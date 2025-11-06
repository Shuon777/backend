import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional,Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from core.relational_service import RelationalService
import json
from infrastructure.llm_integration import get_gigachat
from .geo_service import GeoService
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class SearchService:
    def __init__(
    self, 
    embedding_model_path: str,
    llm_service: Optional[Any] = None,
    species_synonyms_path: Optional[str] = None
):
        """
        Args:
            faiss_index_path: Путь к директории с FAISS индексами
            embedding_model_path: Путь к модели для эмбеддингов
            llm_service: Сервис LLM (опционально, для тестирования)
        """
        self.embedding_model_path = embedding_model_path
        self.llm_service = llm_service or get_gigachat()
        self.relational_service = RelationalService(species_synonyms_path)
        self.geo_service = GeoService()
        self.species_synonyms = self._load_species_synonyms(species_synonyms_path)
        self._build_reverse_synonyms_index()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.object_synonyms = self._load_object_synonyms(species_synonyms_path)
        self._build_reverse_object_synonyms_index()
        
    def _init_gigachat(self):
        if self.llm is None:
            self.llm = get_gigachat()
            
    def _get_llm(self):
        if self.llm_service is None:
            self.llm_service = get_gigachat()
        return self.llm_service
    
    def _load_object_synonyms(self, file_path: Optional[str] = None):
        """Загружает синонимы объектов из JSON файла"""
        # Игнорируем переданный file_path и всегда используем object_synonyms.json
        base_dir = Path(__file__).parent.parent
        file_path = base_dir / "json_files" / "object_synonyms.json"
        
        logger.info(f"Загрузка синонимов объектов из: {file_path}")
        logger.info(f"Файл существует: {file_path.exists()}")
        
        if not file_path.exists():
            logger.error(f"Файл синонимов объектов не найден: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                synonyms = json.load(f)
            logger.info(f"Успешно загружено {len(synonyms)} типов объектов")
            
            # Логируем структуру для отладки
            for obj_type, type_synonyms in synonyms.items():
                if isinstance(type_synonyms, dict):
                    logger.info(f"Тип: {obj_type}, количество записей: {len(type_synonyms)}")
                else:
                    logger.warning(f"Неправильная структура для типа {obj_type}: ожидается dict, получен {type(type_synonyms)}")
                
            return synonyms
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON файла синонимов объектов: {e}")
            return {}
        except Exception as e:
            logger.error(f"Ошибка загрузки синонимов объектов: {e}")
            return {}

        
    def _load_species_synonyms(self, file_path: Optional[str] = None):
        """Загружает синонимы видов из JSON файла"""
        if file_path is None:
            base_dir = Path(__file__).parent.parent
            file_path = base_dir / "json_files" / "species_synonyms.json"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Файл синонимов не найден: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON файла синонимов: {e}")
            return {}
        except Exception as e:
            logger.error(f"Ошибка загрузки синонимов: {e}")
            return {}
        
    def _build_reverse_object_synonyms_index(self):
        """Создает обратный индекс для быстрого поиска по синонимам объектов"""
        logger.info(f"Начало построения индекса синонимов объектов")
        logger.info(f"Загружено типов объектов: {len(self.object_synonyms)}")
        
        self.reverse_object_synonyms = {}
        
        if not self.object_synonyms:
            logger.warning("Нет данных синонимов для построения индекса")
            return
        
        
        # Проверяем структуру object_synonyms
        if not isinstance(self.object_synonyms, dict):
            logger.error(f"object_synonyms должен быть словарем, получен: {type(self.object_synonyms)}")
            return
        
        for object_type, type_synonyms in self.object_synonyms.items():
            # Проверяем, что type_synonyms является словарем
            if not isinstance(type_synonyms, dict):
                logger.warning(f"type_synonyms для типа '{object_type}' должен быть словарем, получен: {type(type_synonyms)}. Пропускаем.")
                continue
                
            for main_name, synonyms in type_synonyms.items():
                # Проверяем, что synonyms является списком
                if not isinstance(synonyms, list):
                    logger.warning(f"synonyms для '{main_name}' должен быть списком, получен: {type(synonyms)}. Пропускаем.")
                    continue
                    
                # Добавляем основную форму в индекс
                normalized_main = main_name.lower()
                if normalized_main not in self.reverse_object_synonyms:
                    self.reverse_object_synonyms[normalized_main] = []
                
                # Проверяем, нет ли уже такой записи
                existing_entry = next((item for item in self.reverse_object_synonyms[normalized_main] 
                                    if item["main_form"] == main_name and item["type"] == object_type), None)
                if not existing_entry:
                    self.reverse_object_synonyms[normalized_main].append({
                        "main_form": main_name,
                        "type": object_type
                    })
                
                # Добавляем все синонимы в индекс
                for synonym in synonyms:
                    normalized_synonym = synonym.lower()
                    if normalized_synonym not in self.reverse_object_synonyms:
                        self.reverse_object_synonyms[normalized_synonym] = []
                    
                    # Проверяем, нет ли уже такой записи для синонима
                    existing_synonym_entry = next((item for item in self.reverse_object_synonyms[normalized_synonym] 
                                                if item["main_form"] == main_name and item["type"] == object_type), None)
                    if not existing_synonym_entry:
                        self.reverse_object_synonyms[normalized_synonym].append({
                            "main_form": main_name,
                            "type": object_type
                        })
                        
    def resolve_object_synonym(self, object_name: str, object_type: str = "all") -> Dict[str, Any]:
        """
        Разрешает синонимы объектов и возвращает основное название и тип
        """
        if not object_name:
            return {"error": "Название объекта не указано"}
        
        try:
            # Если индекс синонимов не построен, возвращаем оригинальное название
            if not hasattr(self, 'reverse_object_synonyms') or not self.reverse_object_synonyms:
                logger.warning("Индекс синонимов объектов не построен")
                return {
                    "main_form": object_name,
                    "object_type": object_type,
                    "original_name": object_name,
                    "resolved": False
                }
            
            normalized_name = object_name.lower()
            
            # Ищем в обратном индексе
            if normalized_name in self.reverse_object_synonyms:
                matches = self.reverse_object_synonyms[normalized_name]
                
                # Если указан конкретный тип, фильтруем по нему
                if object_type != "all":
                    filtered_matches = [m for m in matches if m["type"] == object_type]
                    if filtered_matches:
                        return {
                            "main_form": filtered_matches[0]["main_form"],
                            "object_type": filtered_matches[0]["type"],
                            "original_name": object_name,
                            "resolved": True
                        }
                
                # Если тип не указан или не нашли по указанному типу, берем первый попавшийся
                if matches:
                    return {
                        "main_form": matches[0]["main_form"],
                        "object_type": matches[0]["type"],
                        "original_name": object_name,
                        "resolved": True
                    }
            
            # Проверяем прямое совпадение с основными формами
            if object_type != "all":
                # Ищем в конкретном типе
                if object_type in self.object_synonyms:
                    type_synonyms = self.object_synonyms[object_type]
                    if isinstance(type_synonyms, dict):
                        for main_form, synonyms in type_synonyms.items():
                            if main_form.lower() == normalized_name:
                                return {
                                    "main_form": main_form,
                                    "object_type": object_type,
                                    "original_name": object_name,
                                    "resolved": True
                                }
            else:
                # Ищем во всех типах
                for obj_type, type_synonyms in self.object_synonyms.items():
                    if isinstance(type_synonyms, dict):
                        for main_form, synonyms in type_synonyms.items():
                            if main_form.lower() == normalized_name:
                                return {
                                    "main_form": main_form,
                                    "object_type": obj_type,
                                    "original_name": object_name,
                                    "resolved": True
                                }
            
            # Не нашли синоним
            return {
                "main_form": object_name,
                "object_type": object_type,
                "original_name": object_name,
                "resolved": False
            }
            
        except Exception as e:
            logger.error(f"Ошибка при разрешении синонима объекта '{object_name}': {str(e)}")
            return {
                "main_form": object_name,
                "object_type": object_type,
                "original_name": object_name,
                "resolved": False
            }
        
    def _build_reverse_synonyms_index(self):
        """Создает обратный индекс для быстрого поиска по синонимам"""
        self.reverse_synonyms = {}
        for main_name, synonyms in self.species_synonyms.items():
            for synonym in synonyms:
                normalized_synonym = synonym.lower()
                if normalized_synonym not in self.reverse_synonyms:
                    self.reverse_synonyms[normalized_synonym] = []
                self.reverse_synonyms[normalized_synonym].append(main_name)
                
    def get_synonyms_for_name(self, name: str) -> Dict[str, Any]:
        """
        Возвращает все синонимы для заданного названия
        Args:
            name: Название вида (может быть любым синонимом)
        Returns:
            Словарь с основной формой и всеми синонимами
        """
        if not name:
            return {"error": "Название не указано"}
        normalized_name = name.lower()
        main_forms = self.reverse_synonyms.get(normalized_name, [])
        if not main_forms:
            for main_name, synonyms in self.species_synonyms.items():
                if main_name.lower() == normalized_name:
                    main_forms = [main_name]
                    break
        
        if not main_forms:
            return {"error": f"Название '{name}' не найдено в базе синонимов"}
        
        result = {}
        for main_form in main_forms:
            result[main_form] = self.species_synonyms.get(main_form, [])
        
        return result
    
    def get_object_descriptions(self, object_name: str, object_type: str = "all", in_stoplist: str = "1") -> List[str]:
        """Получает все текстовые описания по названию объекта любого типа с учетом in_stoplist"""
        try:
            all_descriptions = []
            
            # Определяем типы объектов для поиска
            search_types = []
            if object_type == "all":
                search_types = ["biological_entity", "geographical_entity", "modern_human_made","organization","research_project","volunteer_initiative","ancient_human_made"]
            else:
                search_types = [object_type]
            
            for entity_type in search_types:
                # Для всех типов объектов используем реляционный сервис
                descriptions = self.relational_service.get_object_descriptions(object_name, entity_type, in_stoplist=in_stoplist)
                if descriptions:
                    all_descriptions.extend(descriptions)
                        
            return list(set(all_descriptions))
                
        except Exception as e:
            logger.error(f"Ошибка получения описания объекта '{object_name}': {str(e)}")
            return []
    def get_object_descriptions_by_filters(
    self,
    filter_data: Dict[str, Any],
    object_type: str = "all",
    limit: int = 10,
    in_stoplist: str = "1",
    object_name: Optional[str] = None  # Добавляем параметр для точного поиска
) -> List[Dict]:
        """
        Поиск описаний объектов по фильтрам из JSON body с учетом in_stoplist
        и точным поиском по object_name если передан
        """
        try:
            return self.relational_service.get_object_descriptions_by_filters(
                filter_data=filter_data,
                object_type=object_type,
                limit=limit,
                in_stoplist=in_stoplist,
                object_name=object_name  # Передаем object_name для точного поиска
            )
                
        except Exception as e:
            logger.error(f"Ошибка поиска объектов по фильтрам: {str(e)}")
            return []
        
    def get_object_descriptions_with_embedding(self, object_name: str, object_type: str, 
                                        query_embedding: List[float], 
                                        limit: int = 10, 
                                        similarity_threshold: float = 0.05,
                                        in_stoplist: str = "1") -> List[Dict]:
        """Получает текстовые описания объектов с учетом схожести эмбеддингов и in_stoplist"""
        try:
            all_descriptions = []
            
            # Определяем типы объектов для поиска
            search_types = []
            if object_type == "all":
                search_types = ["biological_entity", "geographical_entity", "modern_human_made","organization","research_project","volunteer_initiative","ancient_human_made"]
            else:
                search_types = [object_type]
            
            for entity_type in search_types:
                # Для всех типов объектов используем реляционный сервис
                descriptions = self.relational_service.get_object_descriptions_with_embedding(
                    object_name, entity_type, query_embedding, limit, similarity_threshold, in_stoplist
                )
                if descriptions:
                    all_descriptions.extend(descriptions)
            
            # Сортируем по схожести и ограничиваем количество
            all_descriptions.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            return all_descriptions[:limit]
                
        except Exception as e:
            logger.error(f"Ошибка получения описаний с эмбеддингом для '{object_name}': {str(e)}")
            return []
    def search_objects_by_embedding(
        self, 
        query_embedding: List[float],
        object_type: str = "all",
        limit: int = 10,
        similarity_threshold: float = 0.05,
        in_stoplist: str = "1"
    ) -> List[Dict]:
        """
        Поиск объектов по семантическому сходству с запросом с учетом in_stoplist
        """
        try:
            all_descriptions = []
            
            # Определяем типы объектов для поиска
            search_types = []
            if object_type == "all":
                search_types = ["biological_entity", "geographical_entity", "modern_human_made", 
                            "organization", "research_project", "volunteer_initiative", "ancient_human_made"]
            else:
                search_types = [object_type]
            
            for entity_type in search_types:
                # Используем существующий метод реляционного сервиса для поиска по эмбеддингу
                descriptions = self.relational_service.search_objects_by_embedding_only(
                    query_embedding=query_embedding,
                    object_type=entity_type,
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                    in_stoplist=in_stoplist
                )
                if descriptions:
                    all_descriptions.extend(descriptions)
            
            # Сортируем по схожести и ограничиваем количество
            all_descriptions.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            return all_descriptions[:limit]
                
        except Exception as e:
            logger.error(f"Ошибка семантического поиска объектов: {str(e)}")
            return []
    def _generate_gigachat_answer(self, question: str, context: str) -> Dict[str, Any]:
        """
        Генерирует ответ GigaChat на основе вопроса и контекста
        Возвращает словарь с ответом и метаданными
        """
        llm = self._get_llm()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Ты эксперт по Байкальской природной территории. "
                "Используй твою базу знаний для точных ответов на вопросы пользователя.\n\n"
                "Особые указания:\n"
                "- На вопросы 'сколько' - подсчитай количество соответствующих записей в базе знаний\n"
                "Например, на вопрос 'Сколько музеев?' при информации 'Всего найдено записей: 98 (в контекст включено топ-5 по релевантности)', нужно ответить около 98 музеев и затем описание каждого музея из топ записей"
                "- Будь информативным и лаконичным\n"
                "- Даже при неполной информации предоставь доступные детали\n\n"
                f"Твоя база знаний:\n{context}\n\n"
                f"Вопрос: {question}\n\n"
                "Ответ:"
            ))
        ])
        
        try:
            chain = prompt | llm
            
            # Получаем полный ответ с метаданными
            response = chain.invoke({"question": question, "context": context})
            
            # ДИАГНОСТИКА: Логируем всю структуру ответа
            logger.debug(f"Полный ответ GigaChat: {response}")
            logger.debug(f"Тип ответа: {type(response)}")
            
            # Проверяем различные возможные места для finish_reason
            finish_reason = None
            
            # 1. Проверяем response_metadata
            if hasattr(response, 'response_metadata'):
                logger.debug(f"response_metadata: {response.response_metadata}")
                finish_reason = response.response_metadata.get('finish_reason')
            
            # 2. Проверяем другие возможные атрибуты
            if not finish_reason and hasattr(response, 'llm_output'):
                logger.debug(f"llm_output: {response.llm_output}")
                if isinstance(response.llm_output, dict):
                    finish_reason = response.llm_output.get('finish_reason')
            
            # 3. Проверяем атрибуты самого объекта
            if not finish_reason and hasattr(response, 'finish_reason'):
                finish_reason = response.finish_reason
                
            # 4. Проверяем дополнительные метаданные
            if not finish_reason and hasattr(response, 'additional_kwargs'):
                logger.debug(f"additional_kwargs: {response.additional_kwargs}")
                finish_reason = response.additional_kwargs.get('finish_reason')
            
            logger.debug(f"Найден finish_reason: {finish_reason}")
            
            # Возвращаем словарь с ответом и метаданными
            return {
                "content": response.content.strip() if hasattr(response, 'content') else "",
                "finish_reason": finish_reason,
                "success": finish_reason != 'blacklist'
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа GigaChat: {str(e)}")
            
            # ДИАГНОСТИКА: Логируем полную информацию об ошибке
            logger.debug(f"Тип исключения: {type(e)}")
            if hasattr(e, 'response'):
                logger.debug(f"Response в исключении: {e.response}")
            
            # Проверяем, содержит ли ошибка информацию о blacklist
            error_str = str(e).lower()
            
            return {
                "content": "Извините, не удалось сгенерировать ответ на основе доступной информации.",
                "finish_reason": "error",
                "success": False
            }
    
    
    def search_images_by_features(
    self,
    species_name: str,
    features: Dict[str, Any]
) -> Dict[str, Any]:
        """
        Поиск изображений по названию вида и признакам
        
        Args:
            species_name: Название биологического вида
            features: Словарь с признаками для фильтрации
            
        Returns:
            Результаты поиска изображений
        """
        try:
            synonyms_data = self.get_synonyms_for_name(species_name)
            
            return self.relational_service.search_images_by_features(
                species_name=species_name,
                features=features,
                synonyms_data=synonyms_data
            )
        except Exception as e:
            logger.error(f"Ошибка поиска изображений по признакам: {str(e)}")
            return {
                "status": "error",
                "message": f"Ошибка при поиске изображений: {str(e)}"
            }
            
    def get_text_descriptions(self, species_name: str, in_stoplist: str = "1") -> List[Dict]:
        """Получает все текстовые описания по названию вида с использованием синонимов и учетом in_stoplist"""
        try:
            synonyms_data = self.get_synonyms_for_name(species_name)
            all_descriptions = []
            
            if "error" not in synonyms_data:
                for main_form, synonyms in synonyms_data.items():
                    all_names = [main_form] + synonyms
                    for name in all_names:
                        # Используем реляционный сервис с учетом in_stoplist
                        descriptions = self.relational_service.get_text_descriptions_with_filters(
                            name, in_stoplist=in_stoplist
                        )
                        if descriptions:
                            all_descriptions.extend(descriptions)
            else:
                descriptions = self.relational_service.get_text_descriptions_with_filters(
                    species_name, in_stoplist=in_stoplist
                )
                if descriptions:
                    all_descriptions.extend(descriptions)
                    
            return all_descriptions
                        
        except Exception as e:
            logger.error(f"Ошибка получения описания через RelationalService: {str(e)}")
            return []
        
    def filter_text_descriptions_with_gigachat(self, user_query: str, descriptions: List[Dict]) -> List[Dict]:
        """Фильтрация текстовых описаний видов через GigaChat"""
        llm = self._get_llm()
        
        if not descriptions:
            return []
        descriptions_text = "\n\n".join(
            f"Описание {i+1}:\n{desc.get('content', '')[:800]}..."
            for i, desc in enumerate(descriptions)
        )
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "Ты эксперт по биологическим видам Байкальского региона. Фильтруй текстовые описания СТРОГО по релевантности запросу.\n\n"
                
                "## КРИТЕРИИ ФИЛЬТРАЦИИ:\n"
                "1. Описание ДОЛЖНО полно и точно отвечать на запрос пользователя\n"
                "2. Косвенное упоминание темы = НЕРЕЛЕВАНТНО\n"
                "3. Приоритет отдается описаниям, которые специально посвящены запрашиваемой теме\n\n"
                
                "## ПРИМЕРЫ:\n"
                "Запрос: 'шишка пихты'\n"
                "✓ РЕЛЕВАНТНО: описание, специально посвященное шишкам, их строению, особенностям\n"
                "✗ НЕРЕЛЕВАНТНО: описание, где шишка упоминается вскользь среди другой информации\n\n"
                
                "Запрос: 'кора пихты и питание животных'\n"
                "✓ РЕЛЕВАНТНО: описание о том, какие животные питаются корой пихты\n"
                "✗ НЕРЕЛЕВАНТНО: общее описание пихты с кратким упоминанием коры\n\n"
                
                "## ФОРМАТ ОТВЕТА ТОЛЬКО JSON:\n"
                "{\n"
                "  \"relevant_descriptions\": [список целочисленных индексов релевантных описаний],\n"
                "  \"no_relevant_descriptions\": bool (true если ничего не найдено)\n"
                "}\n\n"
                
                "## ВАЖНО:\n"
                "- Будь строгим в оценке релевантности\n"
                "- Не включай описания, которые лишь косвенно касаются темы\n"
                "- Если нет описаний, полностью отвечающих на запрос, верни пустой список\n"
                "- Возвращай ТОЛЬКО целые числа (например: [0, 2]), НЕ возвращай срезы (например: [0:2])"
            )),
            HumanMessage(content=(
                f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}\n\n"
                f"ДОСТУПНЫЕ ОПИСАНИЯ:\n{descriptions_text}\n\n"
                "ПРОАНАЛИЗИРУЙ и ВЕРНИ JSON ОТВЕТ БЕЗ КОММЕНТАРИЕВ:"
            ))
        ])
        try:
            chain = prompt | llm | JsonOutputParser()
            response = chain.invoke({"user_query": user_query, "descriptions": descriptions_text})
            logger.debug(response)
            if response.get("no_relevant_descriptions", False):
                return []
                
            # Безопасная обработка индексов
            relevant_indices = []
            raw_indices = response.get("relevant_descriptions", [])
            
            logger.debug(f"Raw indices from LLM: {raw_indices}, type: {type(raw_indices)}")
            
            # Обработка различных форматов ответа
            if isinstance(raw_indices, (int, str)):
                raw_indices = [raw_indices]
                
            for idx in raw_indices:
                try:
                    # Если это строка, пытаемся преобразовать в число
                    if isinstance(idx, str):
                        # Обработка строковых представлений срезов
                        if ':' in idx:
                            try:
                                parts = idx.split(':')
                                start = int(parts[0]) if parts[0] else 0
                                stop = int(parts[1]) if parts[1] else len(descriptions)
                                step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
                                slice_indices = list(range(start, stop, step))
                                for slice_idx in slice_indices:
                                    if 0 <= slice_idx < len(descriptions):
                                        relevant_indices.append(slice_idx)
                            except ValueError:
                                continue
                        else:
                            # Обычное число в строке
                            try:
                                num_idx = int(idx)
                                if 0 <= num_idx < len(descriptions):
                                    relevant_indices.append(num_idx)
                            except ValueError:
                                continue
                    # Если это число
                    elif isinstance(idx, int):
                        if 0 <= idx < len(descriptions):
                            relevant_indices.append(idx)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid index '{idx}': {e}")
                    continue
            # Убираем дубликаты и сортируем
            relevant_indices = sorted(set(relevant_indices))
            if not relevant_indices:
                logger.debug("No relevant indices found after processing")
                return []
            logger.debug(f"Processed indices: {relevant_indices}")
            
            return [descriptions[i] for i in relevant_indices]
            
        except Exception as e:
            logger.error(f"Ошибка фильтрации описаний через GigaChat: {str(e)}")
            return descriptions

    def get_objects_in_area_by_type(
    self,
    area_geometry: dict,
    object_type: Optional[str] = None,
    object_subtype: Optional[str] = None,
    object_name: Optional[str] = None,
    limit: int = 70,
    search_around: bool = False,  # Новый параметр
    buffer_radius_km: float = 10.0  # Новый параметр
) -> Dict[str, Any]:
        """
        Поиск объектов в заданной области с фильтрацией по типу и имени
        """
        try:
            # Используем реляционный сервис для выполнения сложного запроса
            results = self.relational_service.get_objects_in_area_by_type(
                area_geometry=area_geometry,
                object_type=object_type,
                object_subtype=object_subtype,
                object_name=object_name,
                limit=limit,
                search_around=search_around,
                buffer_radius_km=buffer_radius_km
            )
            
            if not results:
                # Формируем понятное сообщение в зависимости от критериев поиска
                if object_name:
                    message = f"Объект '{object_name}' не найден в указанной области"
                elif object_type:
                    subtype_msg = f" подтипа '{object_subtype}'" if object_subtype else ""
                    message = f"В указанной области не найдено объектов типа '{object_type}'{subtype_msg}"
                else:
                    message = "В указанной области не найдено объектов"
                    
                return {
                    "answer": message,
                    "objects": [],
                    "area_geometry": area_geometry
                }
            
            # Статистика по расположению объектов
            inside_count = len([obj for obj in results if obj.get('location_type') == 'inside'])
            around_count = len([obj for obj in results if obj.get('location_type') == 'around'])
            
            # Формируем ответное сообщение
            if object_name:
                message = f"Найден объект '{object_name}'"
            else:
                type_msg = f"типа '{object_type}'" if object_type else "всех типов"
                subtype_msg = f" (подтип: {object_subtype})" if object_subtype else ""
                message = f"Найдено {len(results)} объектов {type_msg}{subtype_msg}"
            
            # Добавляем информацию о расположении
            if search_around and around_count > 0:
                location_msg = f" ({inside_count} внутри области, {around_count} в радиусе {buffer_radius_km} км)"
            else:
                location_msg = f" ({inside_count} внутри области)"
            
            message += location_msg
            
            return {
                "answer": message,
                "objects": results,
                "area_geometry": area_geometry,
                "search_stats": {
                    "total": len(results),
                    "inside_area": inside_count,
                    "around_area": around_count,
                    "buffer_radius_km": buffer_radius_km if search_around else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка поиска объектов по типу в области: {str(e)}")
            return {
                "answer": "Ошибка при поиске объектов в области",
                "objects": [],
                "area_geometry": area_geometry
            }
            
    def search_objects_directly_by_name(
    self,
    object_name: str,
    object_type: Optional[str] = None,
    object_subtype: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
        """
        Прямой поиск объектов по имени без привязки к области
        
        Args:
            object_name: Название объекта для поиска
            object_type: Тип объекта (опционально)
            object_subtype: Подтип объекта (опционально)
            limit: Максимальное количество результатов
            
        Returns:
            Результаты поиска объектов
        """
        try:
            # Используем реляционный сервис для поиска объектов по имени
            results = self.relational_service.search_objects_by_name(
                object_name=object_name,
                object_type=object_type,
                object_subtype=object_subtype,
                limit=limit
            )
            
            if not results:
                return {
                    "answer": f"Объект '{object_name}' не найден",
                    "objects": []
                }
            
            return {
                "answer": f"Найден объект '{object_name}'",
                "objects": results
            }
            
        except Exception as e:
            logger.error(f"Ошибка прямого поиска объекта '{object_name}': {str(e)}")
            return {
                "answer": f"Ошибка при поиске объекта '{object_name}'",
                "objects": []
            }
        
    def get_objects_in_polygon(
    self,
    polygon_geojson: dict,
    buffer_radius_km: float = 0,
    object_type: str = None,
    limit: int = 70
) -> Dict[str, Any]:
        """Поиск объектов внутри полигона и в буферной зоне"""
        try:
            results = self.geo_service.get_objects_in_polygon(
                polygon_geojson=polygon_geojson,
                buffer_radius_km=buffer_radius_km,
                object_type=object_type,
                limit=limit
            )
            
            if not results:
                return {
                    "answer": "В указанной области не найдено объектов",
                    "objects": [],
                    "polygon": polygon_geojson,
                    "biological_objects": ""
                }
            
            formatted_results = []
            type_counts = {}
            biological_objects = []
            
            for obj in results:
                obj_type = obj.get("type", "unknown")
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                
                formatted_obj = {
                    "name": obj["name"],
                    "distance": f"{obj['distance_km']:.1f} км от центра",
                    "type": obj_type,
                    "geojson": obj["geojson"]
                }
                
                if obj_type == "biological_entity":
                    biological_objects.append(obj["name"])
                
                if obj.get("description"):
                    formatted_obj["description"] = obj["description"][:200] + "..." if len(obj["description"]) > 200 else obj["description"]
                
                formatted_results.append(formatted_obj)
            
            total_count = len(results)
            type_summary = ", ".join([f"{count} {type_name}" for type_name, count in type_counts.items()])
            area_desc = "полигона" if buffer_radius_km == 0 else f"полигона + {buffer_radius_km}км буфер"
            
            biological_objects_str = ", ".join(biological_objects) if biological_objects else "биологические объекты не найдены"
            
            answer = f"Найдено {total_count} объектов в области {area_desc} ({type_summary}). Биологические объекты: {biological_objects_str}"
            
            return {
                "answer": answer,
                "objects": formatted_results,
                "polygon": polygon_geojson,
                "biological_objects": biological_objects_str 
            }
            
        except Exception as e:
            logger.error(f"Ошибка поиска объектов по полигону: {str(e)}")
            return {
                "answer": "Ошибка при поиске объектов в области",
                "objects": [],
                "polygon": polygon_geojson,
                "biological_objects": ""
            }
            
    def get_nearby_objects(
    self, 
    latitude: float, 
    longitude: float, 
    radius_km: float = 10, 
    limit: int = 20,
    object_type: str = None,
    species_name: Optional[Union[str, List[str]]] = None,
    in_stoplist: int = 1  # Добавить параметр
) -> Dict[str, Any]:
        try:
            # Исправленный вызов GeoService
            start = time.perf_counter()
            results = self.geo_service.get_nearby_objects(
                latitude=latitude,
                longitude=longitude,
                radius_km=radius_km,
                limit=limit,
                object_type=object_type,
                species_name=species_name,
                in_stoplist=in_stoplist  # Передать параметр
            )
            logger.info(f"Nearby objects search took: {time.perf_counter() - start:.2f}s")
            if not results:
                return {
                    "answer": f"В радиусе {radius_km} км не найдено объектов",
                    "objects": []
                }
            # Форматируем результаты
            formatted_results = []
            type_counts = {}
            for obj in results:
                # Собираем статистику по типам
                obj_type = obj.get("type", "unknown")
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                # Форматируем каждый объект
                formatted_obj = {
                    "name": obj["name"],
                    "distance": f"{obj['distance_km']:.1f} км",
                    "type": obj_type,
                    "geojson": obj["geojson"]
                }
                # Добавляем описание, если есть
                if obj.get("description"):
                    formatted_obj["description"] = obj["description"][:200] + "..." if len(obj["description"]) > 200 else obj["description"]
                
                formatted_results.append(formatted_obj)
            # Формируем ответное сообщение
            total_count = len(results)
            type_summary = ", ".join([f"{count} {type_name}" for type_name, count in type_counts.items()])
            answer = f"Найдено {total_count} объектов поблизости ({type_summary})"
            
            return {
                "answer": answer,
                "objects": formatted_results
            }
 
        except Exception as e:
            logger.error(f"Ошибка поиска объектов: {str(e)}")
            return {
                "answer": "Ошибка при поиске ближайших объектов",
                "objects": []
            }
