import logging
from typing import Dict, List, Optional, Tuple
from langchain_core.documents import Document
from collections import defaultdict
from .document_processing import find_resource_by_uri

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_response(search_results: List[Document], 
                  requested_types: Optional[List[str]] = None,
                  scores: Optional[List[float]] = None,
                  debug_mode: bool = False,
                  user_id: Optional[str] = None) -> Tuple[str, Dict[str, List[str]]]:
    """Форматирует результаты поиска в читаемый ответ.
    
    Args:
        search_results: Список найденных документов
        requested_types: Запрошенные типы ресурсов (если есть)
        scores: Список значений схожести для каждого документа (для отладки)
        debug_mode: Флаг отладочного режима
    
    Returns:
        Кортеж из:
        - Форматированный текстовый ответ
        - Словарь с URL ресурсов:
            {
                "images": [список URL изображений],
                "files": [список URL файлов],
                "geo_places": [список географических мест]
            }
    """
    resources = []
    urls = {"images": [], "files": [], "geo_places": []}
    
    special_cases = {
        "изображения": "Изображение",
        "изображение": "Изображение",
        "документы": "Документ",
        "документ": "Документ",
        "аудио": "Аудио",
        "видео": "Видео",
        "графики и диаграммы": "Графики и диаграммы",
        "картографическая": "Картографическая информация",
        "картографическая информация": "Картографическая информация",
        "трансляция": "Трансляция",
        "внешние ссылки": "Внешняя ссылка",
        "ссылки": "Внешняя ссылка",
        "текст": "Текст"
    }

    def extract_description(page_content: str) -> str:
        """Извлекает описание из содержимого документа."""
        if not page_content:
            return ""
        lines = page_content.split('\n')
        for line in lines:
            if line.startswith("Описание:"):
                return line.replace("Описание:", "").strip()
        return page_content
    kart_found = False
    for i, doc in enumerate(search_results):
        meta = doc.metadata
        doc_type = meta.get("type", "Неизвестный тип")
        description = extract_description(doc.page_content)
        resource_URI = meta.get("source", "Unknown")
        file_path = "/var/www/salut_bot/faiss_index_path/resources_dist.json"
        
        resource = find_resource_by_uri(file_path, resource_URI)
        content = ''
        
        if resource:
            doc_type = resource.get("type", doc_type)
            if doc_type == "Картографическая информация":
                if not kart_found:
                    kart_found = True
                    all_synonyms = resource.get('geo_synonyms', [])
                    
                    cleaned_synonyms = []
                    for synonym in all_synonyms:
                        cleaned = synonym.strip().strip('"').strip("'").strip()
                        if cleaned:
                            cleaned_synonyms.append(cleaned)
                    
                    if cleaned_synonyms:
                        urls['geo_places'] = cleaned_synonyms 
                        logger.debug(f'Очищенные гео-синонимы: {cleaned_synonyms}')
            elif doc_type == "Текст":
                content = resource.get('content', '')

        resource_data = {
            "type": doc_type,
            "name": meta.get("name", "Без названия"),
            "description": description,
            "content": content,
            "source": meta.get("file_path", "")
        }

        if debug_mode and scores and i < len(scores):
            resource_data["debug"] = {
                "similarity": round(scores[i], 4),
                "position": i + 1,
                "source_excerpt": doc.page_content[:100] + "..." if doc.page_content else "",
                "metadata": {k: v for k, v in meta.items() if not k.startswith('_')}
            }

        resources.append(resource_data)

        if meta.get("file_path"):
            if doc_type == "Изображение":
                urls["images"].append(meta["file_path"])
            else:
                urls["files"].append(meta["file_path"])

    normalized_requested = []
    if requested_types:
        for t in requested_types:
            normalized_t = special_cases.get(t.lower(), t)
            normalized_requested.append(normalized_t)
        
        filtered_resources = []
        for r in resources:
            if r["type"].lower() in [t.lower() for t in normalized_requested]:
                filtered_resources.append(r)
        resources = filtered_resources

    type_templates = {
      "Изображение": {
    "single": lambda r: (
        f"📷 {r.get('name', 'Без названия')}" +
        (f" - {r['description']}" if (user_id and user_id.startswith("telegram") and r.get('description')) else "") +
        (f"{' (схожесть: ' + str(r['debug']['similarity']) + ')' if debug_mode and 'debug' in r else ''}")
    ).strip(),
    "multiple": lambda rs: (
        "\n".join(
            f"{i+1}. {r.get('name', 'Без названия')}" +
            (f" - {r['description']}" if (user_id and user_id.startswith("telegram") and r.get('description')) else "") +
            (f"{' (схожесть: ' + str(r['debug']['similarity']) + ')' if debug_mode and 'debug' in r else ''}")
            for i, r in enumerate(rs)
        )
    ),
    "none": "Изображения по вашему запросу не найдены"
},

    "Текст": {
          "single": lambda r: (
              f"📝 {r.get('name', 'Текстовый материал')}\n"
              f"{r.get('content', '')}\n"
          ),
        # "multiple": lambda rs: (
        #     "\n".join(
        #         f"{i+1}. {r.get('name', 'Без названия')}\n"
        #         f"{r.get('content', '')}\n"
        #         f"{' (схожесть: ' + str(r['debug']['similarity']) + ')' if debug_mode and 'debug' in r else ''}"
        #         for i, r in enumerate(rs))
            
        # ),
        "multiple": lambda rs: (
            f"📝{rs[0].get('name', 'Без названия')}\n"
            f"{rs[0]['content'] if rs[0].get('content') else ''}"
            f"{' (схожесть: ' + str(rs[0]['debug']['similarity']) + ')' if debug_mode and 'debug' in rs[0] else ''}"
        ),
        "none": "Текстовые материалы по вашему запросу не найдены"
    },
      "Аудио": {
          "single": lambda r: (
              f"🎧 {r.get('name', 'Без названия')}"
              f"{' - ' + r['description'] if r.get('description') else ''}"
              f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
          ).strip(),
          "multiple": lambda rs: (
              "🔊 Несколько аудиофайлов по вашему запросу:\n" +
              "\n".join(
                  f"{i+1}. {r.get('name', 'Без названия')}"
                  f"{' - ' + r['description'] if r.get('description') else ''}"
                  f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
                  for i, r in enumerate(rs))),
          "none": "Аудиофайлы по вашему запросу не найдены"
      },
      "Видео": {
          "single": lambda r: (
              f"🎬 {r.get('name', 'Без названия')}"
              f"{' - ' + r['description'] if r.get('description') else ''}"
              f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
          ).strip(),
          "multiple": lambda rs: (
              "📹 Несколько видео по вашему запросу:\n" +
              "\n".join(
                  f"{i+1}. {r.get('name', 'Без названия')}"
                  f"{' - ' + r['description'] if r.get('description') else ''}"
                  f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
                  for i, r in enumerate(rs))),
          "none": "Видео по вашему запросу не найдены"
      },
      "Документ": {
          "single": lambda r: (
              f"📄 {r.get('name', 'Без названия')}"
              f"{' - ' + r['description'] if r.get('description') else ''}"
              f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
          ).strip(),
          "multiple": lambda rs: (
              "📂 Несколько документов по вашему запросу:\n" +
              "\n".join(
                  f"{i+1}. {r.get('name', 'Без названия')}"
                  f"{' - ' + r['description'] if r.get('description') else ''}"
                  f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
                  for i, r in enumerate(rs))),
          "none": "Документы по вашему запросу не найдены"
      },
      "Графики и диаграммы": {
          "single": lambda r: (
              f"📈 {r.get('name', 'Без названия')}"
              f"{' - ' + r['description'] if r.get('description') else ''}"
              f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
          ).strip(),
          "multiple": lambda rs: (
              "📊 Несколько графиков по вашему запросу:\n" +
              "\n".join(
                  f"{i+1}. {r.get('name', 'Без названия')}"
                  f"{' - ' + r['description'] if r.get('description') else ''}"
                  f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
                  for i, r in enumerate(rs))),
          "none": "Графики по вашему запросу не найдены"
      },
     "Картографическая информация": {
    "single": lambda r: (
        f"🗺️ {r.get('name', 'Без названия')}"
        f"{' - ' + r['content'] if r.get('content') else ''}"
        f"{' (схожесть: ' + str(r['debug']['similarity']) + ')' if debug_mode and 'debug' in r else ''}"
    ).strip(),
    "multiple": lambda rs: (
        f"🗺️ {rs[0].get('name', 'Без названия')}"
        f"{' - ' + rs[0]['content'] if rs[0].get('content') else ''}"
        f"{' (схожесть: ' + str(rs[0]['debug']['similarity']) + ')' if debug_mode and 'debug' in rs[0] else ''}"
        if rs else "Картографические данные по вашему запросу не найдены"
    ),
    "none": "Картографические данные по вашему запросу не найдены"
},

      "Трансляция": {
          "single": lambda r: (
              f"📡 {r.get('name', 'Без названия')}"
              f"{' - ' + r['description'] if r.get('description') else ''}"
              f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
          ).strip(),
          "multiple": lambda rs: (
              "📺 Доступные трансляции:\n" +
              "\n".join(
                  f"{i+1}. {r.get('name', 'Без названия')}"
                  f"{' - ' + r['description'] if r.get('description') else ''}"
                  f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
                  for i, r in enumerate(rs))),
          "none": "Трансляции по вашему запросу не найдены"
      },
      "Внешние ссылки": {
          "single": lambda r: (
              f"🔗 {r.get('name', 'Без названия')}"
              f"{' - ' + r['description'] if r.get('description') else ''}"
              f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
          ).strip(),
          "multiple": lambda rs: (
              "📎 Внешние ресурсы по вашему запросу:\n" +
              "\n".join(
                  f"{i+1}. {r.get('name', 'Без названия')}"
                  f"{' - ' + r['description'] if r.get('description') else ''}"
                  f"{' | Ссылка: ' + r['source'] if r.get('source') else ''}"
                  for i, r in enumerate(rs))),
          "none": "Внешние ресурсы по вашему запросу не найдены"
      }
  }


    resources_by_type = defaultdict(list)
    for res in resources:
        res_type = res['type']
        resources_by_type[res_type].append(res)


    response_parts = []
    known_types = set(type_templates.keys())
    
    for res_type, res_list in resources_by_type.items():
        if res_type in known_types:
            template = type_templates[res_type]
            if len(res_list) == 1 or res_type=='Картографическая информация':
                response = template["single"](res_list[0])
            else:
                response = template["multiple"](res_list)
            #logger.debug(f'Шаблон ответа:{template}')
            response_parts.append(response)

    if requested_types:
        for req_type in requested_types:
            normalized_req_type = special_cases.get(req_type.lower(), req_type)
            if normalized_req_type not in resources_by_type:
                template = type_templates.get(normalized_req_type, {}).get("none")
                
                if template:
                    response_parts.append(template)

    if not response_parts:
        return "По вашему запросу ничего не найдено", urls

    answer = "\n\n".join(response_parts)
    # if(answer=='Картографические данные по вашему запросу не найдены'):
    #     answer='Карта представлена ниже:'
    if debug_mode:
        answer += "\n\n---\nОтладочная информация:\n"
        answer += f"Всего найдено документов: {len(search_results)}\n"
        if scores:
            answer += (
                f"Схожесть: min={min(scores):.2f}, "
                f"max={max(scores):.2f}, "
                f"avg={sum(scores)/len(scores):.2f}\n"
            )
        answer += f"Запрошенные типы: {requested_types or 'Все'}\n"
        answer += f"Найденные типы: {list(resources_by_type.keys())}"
    logger.debug(f'Возвращаемый ответ:{answer}')
    #logger.debug(f'Возвращаемые ссылки:{urls}')
    return answer, urls
