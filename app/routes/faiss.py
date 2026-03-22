import logging
from pathlib import Path
from flask import Blueprint, request, jsonify
from app.services import search_service

faiss_bp = Blueprint('faiss', __name__)
logger = logging.getLogger(__name__)

@faiss_bp.route("/test_faiss_search", methods=["GET"])
def test_faiss_search():
    """
    Тестовый эндпоинт для проверки FAISS векторного поиска
    GET параметры:
    - query: поисковый запрос
    - k: количество результатов (по умолчанию 10)
    - similarity_threshold: порог схожести (0.0-1.0, по умолчанию 0.8)
    - include_full_docs: возвращать полные документы (true/false, по умолчанию false)
    - debug: детальная отладка (true/false, по умолчанию false)
    """
    try:
        query = request.args.get("query", "")
        k = int(request.args.get("k", 10))
        similarity_threshold = float(request.args.get("similarity_threshold", 0.8))
        include_full_docs = request.args.get("include_full_docs", "false").lower() == "true"
        debug = request.args.get("debug", "false").lower() == "true"
        
        if not query:
            return jsonify({
                "status": "error",
                "message": "Параметр 'query' обязателен",
                "example": "/test_faiss_search?query=Байкал&k=5&similarity_threshold=0.7"
            }), 400
        
        logger.info(f"🔍 Тестовый FAISS поиск: '{query}' (k={k}, threshold={similarity_threshold})")
        
        search_service.load_faiss_index()
        
        if not search_service.faiss_vectorstore:
            return jsonify({
                "status": "error",
                "message": "FAISS индекс не загружен",
                "details": f"Проверьте путь: {search_service.faiss_index_path}"
            }), 500
        
        index_size = search_service.faiss_vectorstore.index.ntotal
        logger.info(f"📊 Размер FAISS индекса: {index_size} векторов")
        
        results = search_service.faiss_vectorstore.similarity_search_with_score(query, k=k*2)
        
        logger.info(f"📋 RAW FAISS результаты ({len(results)}):")
        for i, (doc, score) in enumerate(results[:5]):
            score_float = float(score)
            similarity_float = 1 - score_float
            
            logger.info(f"  {i+1}. Score: {score_float:.4f}, Similarity: {similarity_float:.4f}")
            logger.info(f"     Name: {doc.metadata.get('common_name', 'N/A')}")
            logger.info(f"     Type: {doc.metadata.get('resource_type', 'N/A')}")
            logger.info(f"     Resource ID: {doc.metadata.get('resource_id', 'N/A')}")
            
            if debug:
                logger.info(f"     Полный текст чанка: {doc.page_content}")
        
        def convert_floats(obj):
            if isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(item) for item in obj]
            elif hasattr(obj, 'dtype') and 'float32' in str(obj.dtype):
                return float(obj)
            return obj
        
        filtered_results = []
        for doc, score in results:
            score_float = float(score)
            similarity = 1 - score_float
            
            if similarity >= similarity_threshold:
                if include_full_docs:
                    resource_id = doc.metadata.get('resource_id')
                    full_document = search_service._get_full_document(resource_id, doc.page_content)
                    content = full_document
                else:
                    content = doc.page_content
                
                result = {
                    'content': content[:500] + "..." if len(content) > 500 else content,
                    'similarity': similarity,
                    'score': score_float,
                    'metadata': {
                        'resource_id': doc.metadata.get('resource_id', 'unknown'),
                        'resource_type': doc.metadata.get('resource_type', 'unknown'),
                        'common_name': doc.metadata.get('common_name', ''),
                        'scientific_name': doc.metadata.get('scientific_name', ''),
                        'source': doc.metadata.get('source', ''),
                        'chunk_index': doc.metadata.get('chunk_index', 0),
                        'total_chunks': doc.metadata.get('total_chunks', 1)
                    }
                }
                
                if debug:
                    safe_metadata = {}
                    for key, value in doc.metadata.items():
                        if hasattr(value, 'dtype') and 'float32' in str(value.dtype):
                            safe_metadata[key] = float(value)
                        else:
                            safe_metadata[key] = value
                    result['full_metadata'] = safe_metadata
                
                result = convert_floats(result)
                filtered_results.append(result)
        
        filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
        filtered_results = filtered_results[:k]
        
        stats = {
            "total_index_size": index_size,
            "query": query,
            "parameters": {
                "k_requested": k,
                "similarity_threshold": similarity_threshold,
                "include_full_docs": include_full_docs
            },
            "search_results": {
                "raw_results": len(results),
                "filtered_results": len(filtered_results),
                "threshold_passed": len(filtered_results)
            }
        }
        
        resource_types = {}
        for result in filtered_results:
            rtype = result['metadata']['resource_type']
            resource_types[rtype] = resource_types.get(rtype, 0) + 1
        
        stats["resource_types"] = resource_types
        
        response = {
            "status": "success",
            "message": f"Найдено {len(filtered_results)} документов (порог: {similarity_threshold})",
            "stats": stats,
            "results": filtered_results,
            "query": query
        }
        
        if not filtered_results:
            response["status"] = "no_results"
            response["message"] = f"Не найдено документов с порогом схожести {similarity_threshold}"
            response["suggestion"] = "Попробуйте снизить порог similarity_threshold или изменить запрос"
            response["debug_info"] = {
                "raw_results_count": len(results),
                "threshold_applied": similarity_threshold,
                "index_size": index_size
            }
        
        logger.info(f"✅ FAISS поиск завершен: {len(filtered_results)} результатов")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Ошибка в /test_faiss_search: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Ошибка FAISS поиска: {str(e)}",
            "error_details": str(e) if debug else None
        }), 500

@faiss_bp.route("/faiss_status", methods=["GET"])
def faiss_status():
    """Статус FAISS индекса"""
    try:
        status = {
            "faiss_index_path": search_service.faiss_index_path,
            "faiss_vectorstore_loaded": search_service.faiss_vectorstore is not None,
            "resources_by_id_loaded": len(search_service.resources_by_id) > 0,
            "embedding_model_path": search_service.embedding_model_path
        }
        
        if search_service.faiss_vectorstore:
            status["index_size"] = search_service.faiss_vectorstore.index.ntotal
            status["resources_count"] = len(search_service.resources_by_id)
        
        if search_service.faiss_index_path:
            import os
            index_dir = Path(search_service.faiss_index_path)
            if index_dir.exists():
                files = []
                for file in index_dir.glob("*"):
                    size_mb = file.stat().st_size / (1024 * 1024)
                    files.append({
                        "name": file.name,
                        "size_mb": round(size_mb, 2)
                    })
                status["index_files"] = files
            else:
                status["index_files_error"] = f"Директория не найдена: {index_dir}"
        
        return jsonify({
            "status": "success",
            "data": status
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статуса FAISS: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка получения статуса: {str(e)}"
        }), 500