import json
import logging
import math
import time
from pathlib import Path
from flask import Blueprint, request, jsonify
from app.services import search_service
from app.utils import (
    extract_external_id, extract_all_external_ids, get_proper_title, 
    convert_floats, generate_cache_key
)

description_bp = Blueprint('description', __name__)
logger = logging.getLogger(__name__)

@description_bp.route("/object/description/", methods=["GET", "POST"])
def get_object_description():
    logger.info(f"📦 /object/description - GET params: {dict(request.args)}")
    logger.info(f"📦 /object/description - POST data: {request.get_json()}")
    
    object_name = request.args.get("object_name")
    query = request.args.get("query")
    clean_query = request.args.get("clean_query", query)
    limit = int(request.args.get("limit", 1500))
    similarity_threshold = float(request.args.get("similarity_threshold", 0.35))
    include_similarity = request.args.get("include_similarity", "false").lower() == "true"
    use_gigachat_filter = request.args.get("use_gigachat_filter", "false").lower() == "true"
    use_gigachat_answer = request.args.get("use_gigachat_answer", "false").lower() == "true"
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    object_type = request.args.get("object_type", "all")
    save_prompt = request.args.get("save_prompt", "false").lower() == "true"
    in_stoplist = request.args.get("in_stoplist", "1")
    return_raw_documents = request.args.get("return_raw_documents", "false").lower() == "true"
    
    force_vector_search = request.args.get("force_vector_search", "false").lower() == "true"
    vector_similarity_threshold = float(request.args.get("vector_similarity_threshold", "0.03"))
    use_vector_fallback = request.args.get("use_vector_fallback", "true").lower() == "true"

    filter_data = None
    if request.method == "POST" and request.is_json:
        filter_data = request.get_json()
        logger.debug(f"Получены фильтры из body: {filter_data}")

    debug_info = {
        "parameters": {
            "object_name": object_name,
            "object_type": object_type,
            "query": query,
            "clean_query": clean_query,
            "limit": limit,
            "similarity_threshold": similarity_threshold,
            "include_similarity": include_similarity,
            "use_gigachat_filter": use_gigachat_filter,
            "use_gigachat_answer": use_gigachat_answer,
            "filter_data": filter_data,
            "save_prompt": save_prompt,
            "in_stoplist": in_stoplist,
            "force_vector_search": force_vector_search,
            "vector_similarity_threshold": vector_similarity_threshold,
            "use_vector_fallback": use_vector_fallback
        },
        "timestamp": time.time(),
        "steps": []
    }

    resolved_object_info = None
    if object_name:
        resolved_object_info = search_service.resolve_object_synonym(object_name, object_type)
        
        debug_info["synonym_resolution"] = {
            "original_name": object_name,
            "original_type": object_type,
            "resolved_info": resolved_object_info
        }
        
        if resolved_object_info.get("resolved", False):
            object_name = resolved_object_info["main_form"]
            if object_type != "all":
                object_type = resolved_object_info["object_type"]
            logger.info(f"✅ Разрешен синоним объекта: '{resolved_object_info['original_name']}' -> '{object_name}' (тип: {object_type})")
        else:
            logger.info(f"ℹ️ Синоним для объекта '{object_name}' не найден, используем оригинальное название")

    if use_gigachat_answer and not query:
        response = {"error": "Параметр 'query' обязателен при use_gigachat_answer=true"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400

    if not object_name and not query and not filter_data:
        response = {"error": "Необходимо указать object_name, query или передать фильтры в body"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400

    try:
        search_limit = limit if limit > 0 else 1500
        context_limit = 6
        
        search_query = None
        if clean_query:
            search_query = clean_query
        elif query:
            search_query = query
        elif object_name:
            search_query = object_name
        elif filter_data:
            search_query = json.dumps(filter_data, ensure_ascii=False)
        
        use_faiss_fallback = False
        faiss_results = []
        descriptions = []
        
        if force_vector_search and search_query and use_gigachat_answer:
            logger.info(f"🚀 Активирован принудительный FAISS поиск для запроса: {search_query}")
            faiss_results = search_service.vector_search_fallback(
                query=search_query,
                object_type=object_type,
                similarity_threshold=vector_similarity_threshold,
                limit=context_limit
            )
            if faiss_results:
                use_faiss_fallback = True
                descriptions = faiss_results
                debug_info["faiss_search"] = {
                    "activated": True,
                    "reason": "force_vector_search",
                    "query_used": search_query,
                    "results_found": len(faiss_results),
                    "similarity_threshold": vector_similarity_threshold,
                    "search_source": "faiss_vector_store"
                }
            else:
                descriptions = []
                debug_info["faiss_search"] = {
                    "activated": True,
                    "reason": "force_vector_search",
                    "query_used": search_query,
                    "results_found": 0,
                    "search_source": "faiss_vector_store"
                }
        else:
            if filter_data:
                descriptions = search_service.get_object_descriptions_by_filters(
                    filter_data=filter_data,
                    object_type=object_type,
                    limit=search_limit,
                    in_stoplist=in_stoplist,
                    object_name=object_name
                )
                search_method = "filter_search"
            elif query:
                descriptions = search_service.get_object_descriptions_by_filters(
                    filter_data={},
                    object_type=object_type,
                    limit=search_limit,
                    in_stoplist=in_stoplist,
                    object_name=object_name
                ) if object_name else []
                search_method = "text_search"
            else:
                descriptions = search_service.get_object_descriptions(
                    object_name, object_type, in_stoplist=in_stoplist
                )
                search_method = "simple_search"

            if use_vector_fallback and not descriptions and search_query:
                logger.info(f"🔄 Активирован FAISS fallback (нет результатов в реляционной базе): {search_query}")
                faiss_results = search_service.vector_search_fallback(
                    query=search_query,
                    object_type=object_type,
                    similarity_threshold=vector_similarity_threshold,
                    limit=context_limit
                )
                if faiss_results:
                    use_faiss_fallback = True
                    descriptions = faiss_results
                    debug_info["faiss_fallback"] = {
                        "activated": True,
                        "reason": "no_relational_results",
                        "query_used": search_query,
                        "results_found": len(faiss_results),
                        "similarity_threshold": vector_similarity_threshold,
                        "search_source": "faiss_vector_store"
                    }

        safe_descriptions = []
        stoplisted_descriptions = []

        for desc in descriptions:
            if isinstance(desc, dict):
                if use_faiss_fallback:
                    feature_data = desc.get('feature_data', {})
                    if isinstance(feature_data, dict):
                        desc_in_stoplist = feature_data.get('in_stoplist')
                    else:
                        desc_in_stoplist = desc.get('in_stoplist')
                else:
                    feature_data = desc.get("feature_data", {})
                    desc_in_stoplist = feature_data.get("in_stoplist") if feature_data else None
                
                try:
                    requested_level = int(in_stoplist)
                    if desc_in_stoplist is None or int(desc_in_stoplist) <= requested_level:
                        safe_descriptions.append(desc)
                    else:
                        stoplisted_descriptions.append(desc)
                except (ValueError, TypeError):
                    if desc_in_stoplist is None or int(desc_in_stoplist) <= 1:
                        safe_descriptions.append(desc)
                    else:
                        stoplisted_descriptions.append(desc)
            else:
                safe_descriptions.append(desc)

        if debug_mode:
            debug_info["in_stoplist_filter"] = {
                "total_before_filter": len(descriptions),
                "safe_after_filter": len(safe_descriptions),
                "stoplisted_count": len(stoplisted_descriptions),
                "requested_level": in_stoplist
            }

        if not safe_descriptions:
            response = {"error": "Я не готов про это разговаривать"}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 400

        descriptions = safe_descriptions

        if use_gigachat_filter:
            filter_query = query if query else object_name
            if debug_mode:
                debug_info["before_gigachat_filter"] = {
                    "count": len(descriptions),
                    "filter_query": filter_query
                }
            descriptions = search_service.filter_text_descriptions_with_gigachat(filter_query, descriptions)
            if debug_mode:
                debug_info["after_gigachat_filter"] = {
                    "count": len(descriptions),
                    "filtered_out": len(safe_descriptions) - len(descriptions)
                }

        used_objects = []
        not_used_objects = []

        if use_gigachat_answer:
            if not descriptions:
                response = {"error": "Не найдено описаний для генерации ответа"}
                if debug_mode:
                    response["debug"] = debug_info
                return jsonify(response), 404

            safe_descriptions_for_gigachat = []
            blacklisted_descriptions = []
            for desc in descriptions:
                if isinstance(desc, dict):
                    feature_data = desc.get("feature_data", {})
                    if feature_data and feature_data.get("blacklist_risk") is True:
                        blacklisted_descriptions.append(desc)
                        continue
                    if desc.get("blacklist_risk") is True:
                        blacklisted_descriptions.append(desc)
                        continue
                safe_descriptions_for_gigachat.append(desc)

            if debug_mode:
                debug_info["blacklist_filter"] = {
                    "total_before_filter": len(descriptions),
                    "safe_after_filter": len(safe_descriptions_for_gigachat),
                    "blacklisted_count": len(blacklisted_descriptions)
                }

            if not safe_descriptions_for_gigachat:
                response = {"error": "Все описания содержат риск blacklist и не могут быть использованы для генерации ответа GigaChat"}
                if debug_mode:
                    response["debug"] = debug_info
                return jsonify(response), 400

            descriptions_for_context = safe_descriptions_for_gigachat

            if all('similarity' in desc for desc in descriptions_for_context):
                context_descriptions = sorted(descriptions_for_context, key=lambda x: x.get('similarity', 0), reverse=True)[:context_limit]
            else:
                context_descriptions = descriptions_for_context[:context_limit]

            for desc in context_descriptions:
                if isinstance(desc, dict):
                    obj_info = {
                        "name": desc.get("object_name", object_name if object_name else "semantic_search"),
                        "type": desc.get("object_type", object_type),
                        "source": desc.get("source", "unknown"),
                        "similarity": round(desc.get("similarity", 0), 4) if desc.get("similarity") else None,
                        "search_source": "faiss_vector_store" if use_faiss_fallback else "relational_database"
                    }
                    used_objects.append(obj_info)

            remaining_descriptions = [desc for desc in descriptions_for_context if desc not in context_descriptions]
            for desc in remaining_descriptions:
                if isinstance(desc, dict):
                    obj_info = {
                        "name": desc.get("object_name", object_name if object_name else "semantic_search"),
                        "type": desc.get("object_type", object_type),
                        "source": desc.get("source", "unknown"),
                        "similarity": round(desc.get("similarity", 0), 4) if desc.get("similarity") else None,
                        "search_source": "faiss_vector_store" if use_faiss_fallback else "relational_database"
                    }
                    not_used_objects.append(obj_info)

            if return_raw_documents:
                logger.info("📄 Возвращаем сырые документы без вызова GigaChat")
                external_ids = extract_all_external_ids(descriptions_for_context)
                formatted_descriptions = []
                for i, desc in enumerate(descriptions_for_context, 1):
                    if isinstance(desc, dict):
                        content = desc.get("content", "")
                        similarity = desc.get("similarity")
                        source = desc.get("source", "unknown")
                        external_id = extract_external_id(desc)
                        title = get_proper_title(desc, object_name, i)
                        formatted_desc = {
                            "id": i,
                            "title": title,
                            "content": content,
                            "source": source,
                            "feature_data": desc.get("feature_data", {}),
                            "structured_data": desc.get("structured_data", {})
                        }
                        if external_id:
                            formatted_desc["external_id"] = external_id
                        if similarity is not None:
                            formatted_desc["similarity"] = round(similarity, 4)
                        formatted_descriptions.append(formatted_desc)
                    else:
                        formatted_descriptions.append({
                            "id": i,
                            "title": get_proper_title(None, object_name, i),
                            "content": desc,
                            "source": "content"
                        })

                if all('similarity' in desc for desc in formatted_descriptions):
                    formatted_descriptions.sort(key=lambda x: x.get('similarity', 0), reverse=True)

                response_data = {
                    "count": len(formatted_descriptions),
                    "descriptions": formatted_descriptions,
                    "external_id": external_ids,
                    "external_ids": external_ids,
                    "query_used": query if query else "simple_search",
                    "clean_query_used": clean_query if clean_query else None,
                    "similarity_threshold": similarity_threshold if query else None,
                    "use_gigachat_filter": use_gigachat_filter,
                    "use_gigachat_answer": True,
                    "raw_documents": True,
                    "message": "Возвращены исходные документы (GigaChat пропущен по запросу return_raw_documents)",
                    "formatted": True,
                    "in_stoplist_filter_applied": True,
                    "in_stoplist_level": in_stoplist,
                    "used_objects": used_objects,
                    "not_used_objects": not_used_objects
                }

                if object_name:
                    response_data["object_name"] = object_name
                    response_data["object_type"] = object_type

                if filter_data:
                    response_data["filters_applied"] = filter_data

                if resolved_object_info and resolved_object_info.get("resolved", False):
                    response_data["synonym_resolution"] = {
                        "original_name": resolved_object_info["original_name"],
                        "resolved_name": object_name,
                        "original_type": resolved_object_info.get("original_type", object_type)
                    }

                if use_faiss_fallback:
                    response_data["search_source"] = "faiss_vector_store"
                    response_data["vector_similarity_threshold"] = vector_similarity_threshold
                    response_data["faiss_fallback_used"] = True
                    response_data["faiss_search_query"] = search_query
                    response_data["clean_query_for_faiss"] = clean_query if clean_query else None

                if debug_mode:
                    response_data["debug"] = debug_info
                    response_data["debug"]["gigachat_generation"] = {
                        "skipped": True,
                        "reason": "return_raw_documents",
                        "prompt_saved": save_prompt,
                        "external_ids_found": len(external_ids)
                    }

                return jsonify(response_data)

            context = "\n\n".join([
                desc["content"] if isinstance(desc, dict) else desc 
                for desc in context_descriptions
            ])

            total_count = len(descriptions_for_context)
            count_info = f"\n\nВсего найдено безопасных записей: {total_count}"
            if len(blacklisted_descriptions) > 0:
                count_info += f" (исключено {len(blacklisted_descriptions)} записей с риском blacklist)"
            if total_count > context_limit:
                count_info += f" (в контекст включено топ-{context_limit} по релевантности)"
            if use_faiss_fallback:
                count_info += f"\nПоиск выполнен в FAISS векторной базе (порог схожести: {vector_similarity_threshold})"
                if clean_query:
                    count_info += f"\nИспользован очищенный запрос для поиска: '{clean_query}'"

            context += count_info

            full_prompt = f"""Ты эксперт по Байкальской природной территории. 
            Используй твою базу знаний для точных ответов на вопросы пользователя.

            Особые указания:
            - На вопросы 'сколько' - подсчитай количество соответствующих записей в базе знаний
            Например, на вопрос 'Сколько музеев?' при информации 'Всего найдено записей: 98 (в контекст включено топ-5 по релевантности)', нужно ответить около 98 музеев и затем описание каждого музея из топ записей
            - Будь информативным и лаконичным
            - Даже при неполной информации предоставь доступные детали

            Твоя база знаний:
            {context}

            Вопрос: {query}

            Ответ:"""

            if save_prompt:
                current_dir = Path(__file__).parent.parent.parent
                timestamp = int(time.time())
                prompt_filename = current_dir / f"gigachat_prompt_{timestamp}.txt"
                try:
                    with open(prompt_filename, 'w', encoding='utf-8') as f:
                        f.write(full_prompt)
                    logger.info(f"✅ Полный промпт сохранен в: {prompt_filename}")
                except Exception as e:
                    logger.error(f"❌ Ошибка сохранения промпта: {e}")

            try:
                llm_result = search_service._generate_llm_answer(query, context)
                is_blacklist = llm_result.get("finish_reason") == "blacklist" or not llm_result.get("success", True)

                if is_blacklist:
                    logger.info("🚫 GigaChat вернул blacklist, возвращаем форматированные безопасные описания")
                    external_ids = extract_all_external_ids(descriptions_for_context)
                    formatted_descriptions = []
                    for i, desc in enumerate(descriptions_for_context, 1):
                        if isinstance(desc, dict):
                            content = desc.get("content", "")
                            similarity = desc.get("similarity")
                            source = desc.get("source", "unknown")
                            external_id = extract_external_id(desc)
                            title = get_proper_title(desc, object_name, i)
                            formatted_desc = {
                                "id": i,
                                "title": title,
                                "content": content,
                                "source": source,
                                "feature_data": desc.get("feature_data", {}),
                                "structured_data": desc.get("structured_data", {})
                            }
                            if external_id:
                                formatted_desc["external_id"] = external_id
                            if similarity is not None:
                                formatted_desc["similarity"] = round(similarity, 4)
                            formatted_descriptions.append(formatted_desc)
                        else:
                            formatted_descriptions.append({
                                "id": i,
                                "title": get_proper_title(None, object_name, i),
                                "content": desc,
                                "source": "content"
                            })

                    if all('similarity' in desc for desc in formatted_descriptions):
                        formatted_descriptions.sort(key=lambda x: x.get('similarity', 0), reverse=True)

                    response_data = {
                        "count": len(formatted_descriptions),
                        "descriptions": formatted_descriptions,
                        "external_id": external_ids,
                        "external_ids": external_ids,
                        "query_used": query if query else "simple_search",
                        "clean_query_used": clean_query if clean_query else None,
                        "similarity_threshold": similarity_threshold if query else None,
                        "use_gigachat_filter": use_gigachat_filter,
                        "use_gigachat_answer": True,
                        "gigachat_restricted": True,
                        "message": "GigaChat не смог сгенерировать ответ, поэтому возвращены исходные безопасные описания",
                        "formatted": True,
                        "in_stoplist_filter_applied": True,
                        "in_stoplist_level": in_stoplist,
                        "used_objects": used_objects,
                        "not_used_objects": not_used_objects
                    }

                    if object_name:
                        response_data["object_name"] = object_name
                        response_data["object_type"] = object_type

                    if filter_data:
                        response_data["filters_applied"] = filter_data

                    if resolved_object_info and resolved_object_info.get("resolved", False):
                        response_data["synonym_resolution"] = {
                            "original_name": resolved_object_info["original_name"],
                            "resolved_name": object_name,
                            "original_type": resolved_object_info.get("original_type", object_type)
                        }

                    if use_faiss_fallback:
                        response_data["search_source"] = "faiss_vector_store"
                        response_data["vector_similarity_threshold"] = vector_similarity_threshold
                        response_data["faiss_fallback_used"] = True
                        response_data["faiss_search_query"] = search_query
                        response_data["clean_query_for_faiss"] = clean_query if clean_query else None

                    if debug_mode:
                        response_data["debug"] = debug_info
                        response_data["debug"]["gigachat_generation"] = {
                            "finish_reason": llm_result.get("finish_reason"),
                            "blacklist_detected": True,
                            "fallback_to_descriptions": True,
                            "prompt_saved": save_prompt,
                            "external_ids_found": len(external_ids)
                        }

                    return jsonify(response_data)

                gigachat_response = llm_result.get("content", "")
                external_ids = extract_all_external_ids(context_descriptions)
                source_descriptions_summary = []

                for desc in context_descriptions:
                    if isinstance(desc, dict):
                        external_id = extract_external_id(desc)
                        title = get_proper_title(desc, object_name, len(source_descriptions_summary) + 1)
                        desc_summary = {
                            "title": title,
                            "content_preview": desc.get("content", "")[:200] + "..." if len(desc.get("content", "")) > 200 else desc.get("content", ""),
                            "source": desc.get("source", "unknown"),
                            "similarity": round(desc.get("similarity", 0), 4) if desc.get("similarity") else None
                        }
                        if external_id:
                            desc_summary["external_id"] = external_id
                        source_descriptions_summary.append(desc_summary)

                response_data = {
                    "gigachat_answer": gigachat_response,
                    "external_id": external_ids,
                    "external_ids": external_ids,
                    "source_descriptions": source_descriptions_summary,
                    "context_used": {
                        "descriptions_count": len(context_descriptions),
                        "total_descriptions": total_count,
                        "blacklisted_excluded": len(blacklisted_descriptions),
                        "external_ids_count": len(external_ids)
                    },
                    "query": query,
                    "clean_query": clean_query if clean_query else None,
                    "object_name": object_name if object_name else "semantic_search",
                    "object_type": object_type,
                    "in_stoplist_level": in_stoplist,
                    "used_objects": used_objects,
                    "not_used_objects": not_used_objects
                }

                if resolved_object_info and resolved_object_info.get("resolved", False):
                    response_data["synonym_resolution"] = {
                        "original_name": resolved_object_info["original_name"],
                        "resolved_name": object_name,
                        "original_type": resolved_object_info.get("original_type", object_type)
                    }

                if use_faiss_fallback:
                    response_data["search_source"] = "faiss_vector_store"
                    response_data["vector_similarity_threshold"] = vector_similarity_threshold
                    response_data["faiss_fallback_used"] = True
                    response_data["faiss_search_query"] = search_query
                    response_data["clean_query_for_faiss"] = clean_query if clean_query else None

                if debug_mode:
                    response_data["debug"] = debug_info
                    response_data["debug"]["gigachat_generation"] = {
                        "response_length": len(gigachat_response),
                        "finish_reason": llm_result.get("finish_reason"),
                        "blacklist_detected": False,
                        "prompt_saved": save_prompt,
                        "external_ids_found": len(external_ids)
                    }

                response_data = convert_floats(response_data)
                return jsonify(response_data)

            except Exception as e:
                logger.error(f"Ошибка генерации ответа GigaChat: {str(e)}")
                error_response = {"error": "Ошибка генерации ответа GigaChat"}
                if debug_mode:
                    debug_info["gigachat_error"] = str(e)
                    error_response["debug"] = debug_info
                return jsonify(error_response), 500

        for desc in descriptions:
            if isinstance(desc, dict):
                obj_info = {
                    "name": desc.get("object_name", object_name if object_name else "semantic_search"),
                    "type": desc.get("object_type", object_type),
                    "source": desc.get("source", "unknown"),
                    "similarity": round(desc.get("similarity", 0), 4) if desc.get("similarity") else None,
                    "search_source": "faiss_vector_store" if use_faiss_fallback else "relational_database"
                }
                used_objects.append(obj_info)

        if not descriptions:
            response = {"error": "Я не готов про это разговаривать"}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 404

        external_ids = extract_all_external_ids(descriptions)
        formatted_descriptions = []
        for i, desc in enumerate(descriptions, 1):
            if isinstance(desc, dict):
                content = desc.get("content", "")
                similarity = desc.get("similarity")
                source = desc.get("source", "unknown")
                external_id = extract_external_id(desc)
                title = get_proper_title(desc, object_name, i)
                formatted_desc = {
                    "id": i,
                    "title": title,
                    "content": content,
                    "source": source,
                    "feature_data": desc.get("feature_data", {}),
                    "structured_data": desc.get("structured_data", {})
                }
                if external_id:
                    formatted_desc["external_id"] = external_id
                if similarity is not None:
                    formatted_desc["similarity"] = round(similarity, 4)
                formatted_descriptions.append(formatted_desc)
            else:
                formatted_descriptions.append({
                    "id": i,
                    "title": get_proper_title(None, object_name, i),
                    "content": desc,
                    "source": "content"
                })

        if all('similarity' in desc for desc in formatted_descriptions):
            formatted_descriptions.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        response_data = {
            "count": len(formatted_descriptions),
            "descriptions": formatted_descriptions,
            "external_id": external_ids,
            "external_ids": external_ids,
            "query_used": query if query else "simple_search",
            "clean_query_used": clean_query if clean_query else None,
            "similarity_threshold": similarity_threshold if query else None,
            "use_gigachat_filter": use_gigachat_filter,
            "in_stoplist_filter_applied": True,
            "in_stoplist_level": in_stoplist,
            "formatted": True,
            "used_objects": used_objects,
            "not_used_objects": []
        }

        if object_name:
            response_data["object_name"] = object_name
            response_data["object_type"] = object_type

        if filter_data:
            response_data["filters_applied"] = filter_data

        if resolved_object_info and resolved_object_info.get("resolved", False):
            response_data["synonym_resolution"] = {
                "original_name": resolved_object_info["original_name"],
                "resolved_name": object_name,
                "original_type": resolved_object_info.get("original_type", object_type)
            }

        if use_faiss_fallback:
            response_data["search_source"] = "faiss_vector_store"
            response_data["vector_similarity_threshold"] = vector_similarity_threshold
            response_data["faiss_fallback_used"] = True
            response_data["faiss_search_query"] = search_query
            response_data["clean_query_for_faiss"] = clean_query if clean_query else None

        if debug_mode:
            response_data["debug"] = debug_info
            response_data["debug"]["external_ids_extracted"] = {
                "count": len(external_ids),
                "ids": external_ids
            }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Ошибка получения описания: {str(e)}", exc_info=True)
        error_response = {"error": "Внутренняя ошибка сервера"}
        if debug_mode:
            debug_info["error"] = str(e)
            error_response["debug"] = debug_info
        return jsonify(error_response), 500

@description_bp.route("/species/description/", methods=["GET"])
def get_species_description():
    logger.info(f"📦 /species/description - GET params: {dict(request.args)}")
    species_name = request.args.get("species_name")
    query = request.args.get("query")
    limit = int(request.args.get("limit", 1500))
    similarity_threshold = float(request.args.get("similarity_threshold", 0.1))
    include_similarity = request.args.get("include_similarity", "false").lower() == "true"
    use_gigachat_filter = request.args.get("use_gigachat_filter", "false").lower() == "true"
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    in_stoplist = request.args.get("in_stoplist", "1")
    
    force_vector_search = request.args.get("force_vector_search", "false").lower() == "true"
    vector_similarity_threshold = float(request.args.get("vector_similarity_threshold", "0.03"))
    use_vector_fallback = request.args.get("use_vector_fallback", "true").lower() == "true"

    if not species_name:
        response = {
            "error": "species_name parameter is required",
            "used_objects": [],
            "not_used_objects": []
        }
        return jsonify(response), 400

    debug_info = {
        "parameters": {
            "species_name": species_name,
            "query": query,
            "limit": limit,
            "similarity_threshold": similarity_threshold,
            "include_similarity": include_similarity,
            "use_gigachat_filter": use_gigachat_filter,
            "in_stoplist": in_stoplist,
            "force_vector_search": force_vector_search,
            "vector_similarity_threshold": vector_similarity_threshold,
            "use_vector_fallback": use_vector_fallback
        },
        "timestamp": time.time()
    }

    try:
        search_query = query if query else species_name
        
        use_faiss_fallback = False
        descriptions = []
        
        if force_vector_search and query:
            logger.info(f"🚀 Активирован принудительный FAISS поиск для вида: {species_name}, запрос: {query}")
            faiss_results = search_service.vector_search_fallback(
                query=query,
                object_type="biological_entity",
                similarity_threshold=vector_similarity_threshold,
                limit=limit
            )
            if faiss_results:
                use_faiss_fallback = True
                descriptions = faiss_results
                debug_info["faiss_search"] = {
                    "activated": True,
                    "reason": "force_vector_search",
                    "query_used": query,
                    "results_found": len(faiss_results),
                    "similarity_threshold": vector_similarity_threshold,
                    "search_source": "faiss_vector_store"
                }
        else:
            if query:
                descriptions = search_service.get_text_descriptions(species_name, in_stoplist=in_stoplist)
                
                if use_vector_fallback and not descriptions:
                    logger.info(f"🔄 Активирован FAISS fallback для вида: {species_name}, запрос: {query}")
                    faiss_results = search_service.vector_search_fallback(
                        query=query,
                        object_type="biological_entity",
                        similarity_threshold=vector_similarity_threshold,
                        limit=limit
                    )
                    if faiss_results:
                        use_faiss_fallback = True
                        descriptions = faiss_results
                        debug_info["faiss_fallback"] = {
                            "activated": True,
                            "reason": "no_relational_results",
                            "query_used": query,
                            "results_found": len(faiss_results),
                            "similarity_threshold": vector_similarity_threshold,
                            "search_source": "faiss_vector_store"
                        }
            else:
                descriptions = search_service.get_text_descriptions(species_name, in_stoplist=in_stoplist)

        safe_descriptions = []
        stoplisted_descriptions = []
        
        logger.info(f"🔒 ФИЛЬТРАЦИЯ ПО STOPLIST (уровень {in_stoplist}):")
        logger.info(f"   - Всего описаний до фильтрации: {len(descriptions)}")
        
        for desc in descriptions:
            if isinstance(desc, dict):
                if use_faiss_fallback:
                    feature_data = desc.get('feature_data', {})
                    if isinstance(feature_data, dict):
                        desc_in_stoplist = feature_data.get('in_stoplist')
                    else:
                        desc_in_stoplist = desc.get('in_stoplist')
                else:
                    feature_data = desc.get("feature_data", {})
                    desc_in_stoplist = feature_data.get("in_stoplist") if feature_data else None
                
                try:
                    requested_level = int(in_stoplist)
                    if desc_in_stoplist is None or int(desc_in_stoplist) <= requested_level:
                        safe_descriptions.append(desc)
                        logger.info(f"   ✓ БЕЗОПАСНО: in_stoplist={desc_in_stoplist}")
                    else:
                        stoplisted_descriptions.append(desc)
                        logger.info(f"   ✗ STOPLIST: in_stoplist={desc_in_stoplist} > запрошенного {requested_level}")
                except (ValueError, TypeError):
                    if desc_in_stoplist is None or int(desc_in_stoplist) <= 1:
                        safe_descriptions.append(desc)
                        logger.info(f"   ✓ БЕЗОПАСНО (по умолчанию): in_stoplist={desc_in_stoplist}")
                    else:
                        stoplisted_descriptions.append(desc)
                        logger.info(f"   ✗ STOPLIST (по умолчанию): in_stoplist={desc_in_stoplist}")
            else:
                safe_descriptions.append(desc)
                logger.info(f"   ✓ БЕЗОПАСНО: простое описание")

        if debug_mode:
            debug_info["in_stoplist_filter"] = {
                "total_before_filter": len(descriptions),
                "safe_after_filter": len(safe_descriptions),
                "stoplisted_count": len(stoplisted_descriptions),
                "requested_level": in_stoplist
            }

        logger.info(f"📋 ИТОГИ ФИЛЬТРАЦИИ:")
        logger.info(f"   - Безопасные описания: {len(safe_descriptions)}")
        logger.info(f"   - Исключено по stoplist: {len(stoplisted_descriptions)}")

        if not safe_descriptions:
            logger.warning(f"🚫 НЕТ БЕЗОПАСНЫХ ОПИСАНИЙ для '{species_name}'")
            response = {
                "error": "Я не готов про это разговаривать",
                "used_objects": [],
                "not_used_objects": []
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 400

        descriptions = safe_descriptions

        used_objects = []
        not_used_objects = []

        for desc in descriptions:
            similarity = None
            if isinstance(desc, dict) and desc.get("similarity") is not None:
                try:
                    similarity_val = desc.get("similarity")
                    if similarity_val is not None and not math.isnan(float(similarity_val)):
                        similarity = round(float(similarity_val), 4)
                except (ValueError, TypeError):
                    similarity = None
            
            object_name = desc.get("object_name", species_name) if isinstance(desc, dict) else species_name
            
            used_objects.append({
                "name": object_name,
                "type": "biological_entity",
                "source": desc.get("source", "unknown") if isinstance(desc, dict) else "content",
                "similarity": similarity,
                "search_source": "faiss_vector_store" if use_faiss_fallback else "relational_database"
            })

        if use_gigachat_filter:
            filter_query = query if query else species_name
            
            if debug_mode:
                debug_info["before_gigachat_filter"] = {
                    "count": len(descriptions),
                    "filter_query": filter_query
                }
            
            filtered_descriptions = search_service.filter_text_descriptions_with_gigachat(
                filter_query, 
                descriptions
            )
            
            if debug_mode:
                debug_info["after_gigachat_filter"] = {
                    "count": len(filtered_descriptions),
                    "filtered_out": len(descriptions) - len(filtered_descriptions)
                }

            if filtered_descriptions:
                used_objects = []
                for desc in filtered_descriptions:
                    object_name = desc.get("object_name", species_name) if isinstance(desc, dict) else species_name
                    used_objects.append({
                        "name": object_name,
                        "type": "biological_entity", 
                        "source": desc.get("source", "unknown") if isinstance(desc, dict) else "content",
                        "similarity": round(desc.get("similarity", 0), 4) if isinstance(desc, dict) and desc.get("similarity") else None,
                        "search_source": "faiss_vector_store" if use_faiss_fallback else "relational_database"
                    })

            descriptions = filtered_descriptions

        if not descriptions:
            logger.warning(f"🚫 ОПИСАНИЯ ОТФИЛЬТРОВАНЫ GigaChat для '{species_name}'")
            response = {
                "error": "Я не готов про это разговаривать",
                "used_objects": [],
                "not_used_objects": []
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 404

        def format_content_with_title(desc, index):
            if isinstance(desc, dict):
                content = desc.get("content", "")
                object_name = desc.get("object_name", species_name)
                
                title_header = f"** {object_name} **\n\n"
                formatted_content = title_header + content
                return formatted_content
            else:
                return f"# {species_name}\n\n{desc}"
        
        if include_similarity:
            formatted_descriptions = []
            for i, desc in enumerate(descriptions, 1):
                if isinstance(desc, dict):
                    formatted_desc = {
                        "content": format_content_with_title(desc, i),
                        "source": desc.get("source", "unknown"),
                        "feature_data": desc.get("feature_data", {}),
                        "object_name": desc.get("object_name", species_name),
                        "object_type": "biological_entity",
                        "similarity": round(desc.get("similarity", 0), 4) if desc.get("similarity") else None
                    }
                    
                    if desc.get("structured_data"):
                        formatted_desc["structured_data"] = desc.get("structured_data")
                    
                    if desc.get("species_features"):
                        formatted_desc["species_features"] = desc.get("species_features")
                        
                    formatted_descriptions.append(formatted_desc)
                else:
                    formatted_descriptions.append({
                        "content": format_content_with_title(desc, i),
                        "source": "content",
                        "object_name": species_name,
                        "object_type": "biological_entity"
                    })
        else:
            formatted_descriptions = []
            for i, desc in enumerate(descriptions, 1):
                if isinstance(desc, dict):
                    formatted_desc = {
                        "content": format_content_with_title(desc, i),
                        "source": desc.get("source", "unknown"),
                        "feature_data": desc.get("feature_data", {}),
                        "object_name": desc.get("object_name", species_name),
                        "object_type": "biological_entity"
                    }
                    
                    if desc.get("structured_data"):
                        formatted_desc["structured_data"] = desc.get("structured_data")
                    
                    if desc.get("species_features"):
                        formatted_desc["species_features"] = desc.get("species_features")
                        
                    formatted_descriptions.append(formatted_desc)
                else:
                    formatted_descriptions.append({
                        "content": format_content_with_title(desc, i),
                        "source": "content",
                        "object_name": species_name,
                        "object_type": "biological_entity"
                    })

        response_data = {
            "count": len(formatted_descriptions),
            "descriptions": formatted_descriptions,
            "query_used": query if query else "simple_search",
            "similarity_threshold": similarity_threshold if query else None,
            "use_gigachat_filter": use_gigachat_filter,
            "in_stoplist_filter_applied": True,
            "in_stoplist_level": in_stoplist,
            "used_objects": used_objects,
            "not_used_objects": not_used_objects
        }

        if use_faiss_fallback:
            response_data["search_source"] = "faiss_vector_store"
            response_data["vector_similarity_threshold"] = vector_similarity_threshold
            response_data["faiss_fallback_used"] = True
            response_data["faiss_search_query"] = query

        if debug_mode:
            response_data["debug"] = debug_info

        logger.info(f"✅ УСПЕШНЫЙ ОТВЕТ для '{species_name}': {len(formatted_descriptions)} описаний")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Ошибка получения описания для '{species_name}': {str(e)}", exc_info=True)
        error_response = {
            "error": "Внутренняя ошибка сервера",
            "used_objects": [],
            "not_used_objects": []
        }
        if debug_mode:
            debug_info["error"] = str(e)
            error_response["debug"] = debug_info
        return jsonify(error_response), 500