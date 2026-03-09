import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
import json
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from shapely.geometry import shape, mapping

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    
class SearchService:
    def __init__(
        self, 
        embedding_model_path: str,
        llm_service: Optional[Any] = None,
        faiss_index_path: Optional[str] = None
    ):
        """
        Args:
            faiss_index_path: Путь к директории с FAISS индексами
            embedding_model_path: Путь к модели для эмбеддингов
            llm_service: Сервис LLM (опционально, для тестирования)
        """
        self.embedding_model_path = embedding_model_path
        self.llm_service = llm_service
        self.relational_service = None  # Будет инициализирован позже
        self.geo_service = None  # Будет инициализирован позже
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.reranker_model_name = "DiTy/cross-encoder-russian-msmarco"
        self.reranker_local_path = str(Path(__file__).parent.parent / "embedding_models" / "rerankers" / "DiTy_cross-encoder-russian-msmarco")
        self.reranker = None
        
        self.object_synonyms = {}
        self.reverse_object_synonyms = {}
        
        # FAISS атрибуты
        self.faiss_index_path = faiss_index_path
        self.faiss_vectorstore = None
        self.resources_data = None
        self.resources_by_id = {}
        
        # Инициализация зависимостей
        self._init_dependencies()
        self._load_reranker()
        
        # Загрузка данных для FAISS
        if faiss_index_path:
            self._load_resources_data()
            
    def _load_reranker(self):
        """Ленивая загрузка модели реранкера (только при наличии GPU)"""
        if self.reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                # Проверяем наличие CUDA
                cuda_available = False
                if TORCH_AVAILABLE:
                    cuda_available = torch.cuda.is_available()
                    if cuda_available:
                        logger.info("🔄 CUDA доступна, реранкер будет загружен на GPU")
                    else:
                        logger.info("⚠️ CUDA не обнаружена, реранкер отключён (используется только FAISS)")
                        return None
                else:
                    logger.info("⚠️ Torch не установлен, реранкер отключён")
                    return None

                # Загружаем модель на GPU
                logger.info(f"🔄 Загрузка модели реранкера: {self.reranker_model_name}")
                self.reranker = CrossEncoder(self.reranker_local_path, device='cuda')
                logger.info("✅ Модель реранкера загружена на GPU")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки реранкера: {str(e)}")
                self.reranker = None
        return self.reranker

    def _init_dependencies(self):
        """Инициализирует зависимости чтобы избежать циклического импорта"""
        from core.relational_service import RelationalService
        from .geo_service import GeoService
        
        self.relational_service = RelationalService()
        self.geo_service = GeoService()
        
        # Загружаем синонимы
        self.object_synonyms = self._load_object_synonyms()
        self._build_reverse_object_synonyms_index()
    
    def _load_resources_data(self):
        """Загружает данные из resources_dist.json для сопоставления resource_id с полными документами"""
        try:
            # Ищем файл resources_dist.json в разных местах
            base_dir = Path(__file__).parent.parent.parent
            possible_paths = [
                base_dir / "json_files" / "resources_dist.json",
                base_dir.parent / "json_files" / "resources_dist.json",
                Path.cwd() / "json_files" / "resources_dist.json",
                Path("/json_files") / "resources_dist.json"
            ]
            
            resources_file = None
            for path in possible_paths:
                if path.exists():
                    resources_file = path
                    break
            
            if not resources_file:
                logger.warning("Файл resources_dist.json не найден")
                return
            
            logger.info(f"Загружаем ресурсы из: {resources_file}")
            
            with open(resources_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.resources_data = data
            
            # Создаем словарь для быстрого поиска по resource_id
            for resource in data.get('resources', []):
                resource_id = resource.get('identificator', {}).get('id')
                if resource_id:
                    self.resources_by_id[resource_id] = resource
            
            logger.info(f"Загружено {len(self.resources_by_id)} ресурсов для FAISS поиска")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки resources_dist.json: {str(e)}")
    
    def load_faiss_index(self):
        """Загружает FAISS индекс если он еще не загружен"""
        if self.faiss_vectorstore is not None:
            return self.faiss_vectorstore
        
        if not self.faiss_index_path:
            logger.warning("Путь к FAISS индексу не указан")
            return None
        
        try:
            from langchain_community.vectorstores import FAISS
            
            logger.info(f"Загружаем FAISS индекс из {self.faiss_index_path}")
            
            if not os.path.exists(self.faiss_index_path):
                logger.error(f"Директория с FAISS индексом не найдена: {self.faiss_index_path}")
                return None
            
            self.faiss_vectorstore = FAISS.load_local(
                self.faiss_index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"FAISS индекс загружен, содержит {self.faiss_vectorstore.index.ntotal} векторов")
            return self.faiss_vectorstore
            
        except Exception as e:
            logger.error(f"Ошибка загрузки FAISS индекса: {str(e)}")
            return None
    
    def search_in_faiss(self, query: str, k: int = 20, similarity_threshold: float = 0.5) -> List[Dict]:
        """
        Выполняет поиск в FAISS индексе по запросу с использованием inner product
        
        Args:
            query: Поисковый запрос
            k: Количество результатов из FAISS (top-k)
            similarity_threshold: Порог схожести (0.0-1.0)
            
        Returns:
            Список найденных документов с метаданными
        """
        # Загружаем FAISS индекс если нужно
        if not self.faiss_vectorstore:
            if not self.load_faiss_index():
                logger.warning("FAISS индекс не загружен")
                return []
        
        try:
            # Выполняем поиск в FAISS с inner product (чем больше score, тем лучше)
            results = self.faiss_vectorstore.similarity_search_with_score(query, k=k)
            
            # Фильтруем по порогу схожести и преобразуем в нужный формат
            filtered_results = []
            
            for doc, score in results:
                # Для inner product с нормализованными векторами score = косинусное сходство
                similarity = float(score)  # Преобразуем numpy.float32 в float
                
                # Логируем для отладки
                logger.debug(f"FAISS результат: similarity={similarity:.4f}, порог={similarity_threshold}")
                
                if similarity >= similarity_threshold:
                    # Извлекаем полный документ по resource_id из метаданных
                    resource_id = doc.metadata.get('resource_id')
                    full_document = self._get_full_document(resource_id, doc.page_content)
                    
                    if full_document:
                        # Получаем информацию о чанке
                        chunk_index = doc.metadata.get('chunk_index', 0)
                        total_chunks = doc.metadata.get('total_chunks', 1)
                        
                        # Формируем результат
                        result = {
                            'content': full_document,
                            'similarity': similarity,
                            'source': 'faiss_vector_search',
                            'object_name': doc.metadata.get('common_name', ''),
                            'object_type': self._normalize_object_type(doc.metadata.get('resource_type', 'unknown')),
                            'feature_data': {
                                'in_stoplist': doc.metadata.get('in_stoplist', 1),
                                'source': doc.metadata.get('source', '')
                            },
                            'metadata': {
                                'resource_id': resource_id,
                                'chunk_index': chunk_index,
                                'total_chunks': total_chunks,
                                'common_name': doc.metadata.get('common_name', ''),
                                'scientific_name': doc.metadata.get('scientific_name', ''),
                                'resource_type': doc.metadata.get('resource_type', 'unknown')
                            }
                        }
                        
                        # Добавляем информацию о том, что это часть документа
                        if total_chunks > 1:
                            result['chunk_info'] = f"Часть {chunk_index + 1} из {total_chunks}"
                        
                        filtered_results.append(result)
            
            # Сортируем по убыванию схожести (inner product уже отсортирован, но на всякий случай)
            filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"✅ FAISS поиск: найдено {len(filtered_results)} документов из {len(results)} с порогом {similarity_threshold}")
            
            # Логируем первые несколько результатов для отладки
            for i, res in enumerate(filtered_results[:10]):
                logger.debug(f"  {i+1}. {res['object_name']} (схожесть: {res['similarity']:.4f})")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска в FAISS: {str(e)}", exc_info=True)
            return []
        
    def _normalize_object_type(self, object_type: str) -> str:
        """Нормализует тип объекта для совместимости с существующей системой"""
        if not object_type:
            return "unknown"
        
        type_mapping = {
            'Текст': 'biological_entity',
            'Географический объект': 'geographical_entity',
            'Объект флоры': 'biological_entity',
            'Объект фауны': 'biological_entity',
            'biological_entity': 'biological_entity',
            'geographical_entity': 'geographical_entity',
            'modern_human_made': 'modern_human_made',
            'ancient_human_made': 'ancient_human_made',
            'organization': 'organization',
            'research_project': 'research_project',
            'volunteer_initiative': 'volunteer_initiative'
        }
        
        return type_mapping.get(object_type, object_type.lower())
    
    def _get_full_document(self, resource_id: str, chunk_content: str) -> str:
        """
        Получает полный документ по resource_id или возвращает чанк как есть
        
        Args:
            resource_id: ID ресурса из метаданных
            chunk_content: Содержимое чанка
            
        Returns:
            Полный текст документа
        """
        if not resource_id or not self.resources_by_id:
            return chunk_content
        
        try:
            # Ищем ресурс по ID
            resource = self.resources_by_id.get(resource_id)
            if not resource:
                logger.debug(f"Ресурс с ID {resource_id} не найден в базе данных")
                return chunk_content
            
            # Извлекаем полный текст в зависимости от типа ресурса
            resource_type = resource.get('type')
            
            if resource_type == 'Текст':
                # Для текстовых ресурсов используем content
                content = resource.get('content', '')
                if content:
                    return content
                
                # Или извлекаем из structured_data если нет content
                structured_data = resource.get('structured_data', {})
                if structured_data:
                    # Конвертируем structured_data в текстовый формат
                    return self._convert_structured_data_to_text(structured_data)
            
            elif resource_type == 'Географический объект':
                # Для географических объектов используем описание
                description = resource.get('description', '')
                common_name = resource.get('identificator', {}).get('name', {}).get('common', '')
                
                full_text = []
                if common_name:
                    full_text.append(common_name)
                if description:
                    full_text.append(description)
                
                return " ".join(full_text) if full_text else chunk_content
            
            # Возвращаем чанк если не смогли получить полный документ
            return chunk_content
            
        except Exception as e:
            logger.error(f"Ошибка получения полного документа для {resource_id}: {str(e)}")
            return chunk_content
    
    def _convert_structured_data_to_text(self, structured_data: Dict) -> str:
        """
        Конвертирует structured_data в читаемый текст
        
        Args:
            structured_data: Структурированные данные
            
        Returns:
            Текстовое представление
        """
        if not structured_data:
            return ""
        
        sections = []
        
        for section_name, section_data in structured_data.items():
            if isinstance(section_data, dict):
                section_text = [f"{section_name}:"]
                for key, value in section_data.items():
                    if value and str(value).strip() and str(value) != '-':
                        section_text.append(f"  {key}: {value}")
                
                if len(section_text) > 1:
                    sections.append("\n".join(section_text))
        
        return "\n\n".join(sections)
    
    def vector_search_fallback(self, query: str, object_type: str = "all", 
                           similarity_threshold: float = 0.5, limit: int = 5,
                           rerank_top_k: int = 10) -> List[Dict]:
        """
        Fallback метод для векторного поиска с реранком (как в ноутбуке)
        
        Args:
            query: Поисковый запрос
            object_type: Тип объекта для фильтрации
            similarity_threshold: Порог схожести для FAISS
            limit: Количество результатов после реранка (top-5)
            rerank_top_k: Количество результатов из FAISS для реранка (top-20)
            
        Returns:
            Список найденных документов после реранка
        """
        # 1. Получаем топ-k из FAISS
        faiss_results = self.search_in_faiss(
            query=query, 
            k=rerank_top_k, 
            similarity_threshold=similarity_threshold
        )
        
        if not faiss_results:
            logger.info("❌ FAISS fallback: нет результатов")
            return []
        
        logger.info(f"🔍 FAISS fallback: получено {len(faiss_results)} результатов для реранка")
        
        # 2. Применяем реранкер как в ноутбуке
        try:
            reranker = self._load_reranker()
            if reranker:
                # Подготавливаем пары [запрос, текст] для реранкера
                pairs = [[query, r['content']] for r in faiss_results]
                
                # Получаем оценки реранкера
                rerank_scores = reranker.predict(pairs, batch_size=32)
                
                # Сортируем по убыванию скоров реранкера
                reranked = sorted(zip(faiss_results, rerank_scores), 
                                key=lambda x: x[1], reverse=True)
                
                # Извлекаем результаты
                reranked_results = [r for r, _ in reranked]
                
                logger.info(f"✅ Реранкер применил оценки к {len(reranked_results)} документам")
                
                # Логируем изменения в топе
                if len(faiss_results) > 0 and len(reranked_results) > 0:
                    logger.debug(f"  До реранка: {faiss_results[0]['object_name']} ({faiss_results[0]['similarity']:.4f})")
                    logger.debug(f"  После реранка: {reranked_results[0]['object_name']}")
            else:
                logger.warning("⚠️ Реранкер не загружен, используем результаты FAISS без реранка")
                reranked_results = faiss_results
                
        except Exception as e:
            logger.error(f"❌ Ошибка при реранке: {str(e)}")
            # Если реранкер не сработал, используем исходные результаты
            reranked_results = faiss_results
        
        # 3. Фильтруем по типу объекта если указан
        if object_type != "all":
            target_type = self._normalize_object_type(object_type)
            filtered_by_type = []
            
            for result in reranked_results:
                result_type = result.get('object_type', '').lower()
                if result_type == target_type.lower():
                    filtered_by_type.append(result)
                elif target_type.lower() == 'biological_entity' and result_type in ['biological_entity', 'объект флоры', 'объект фауны']:
                    filtered_by_type.append(result)
                elif target_type.lower() == 'geographical_entity' and result_type in ['geographical_entity', 'географический объект']:
                    filtered_by_type.append(result)
            
            reranked_results = filtered_by_type
            logger.info(f"📊 После фильтрации по типу '{object_type}': {len(reranked_results)} результатов")
        
        # 4. Возвращаем топ-limit
        final_results = reranked_results[:limit]
        
        logger.info(f"🎯 FAISS fallback: возвращаем {len(final_results)} результатов (порог: {similarity_threshold}, реранк: {rerank_top_k}->{limit})")
        
        return final_results

    def _load_object_synonyms(self):
        """Загружает синонимы объектов из JSON файла"""
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
    
    def _build_reverse_object_synonyms_index(self):
        """Создает обратный индекс для быстрого поиска по синонимам объектов"""
        logger.info(f"Начало построения индекса синонимов объектов")
        logger.info(f"Загружено типов объектов: {len(self.object_synonyms)}")
        
        self.reverse_object_synonyms = {}
        
        if not self.object_synonyms:
            logger.warning("Нет данных синонимов для построения индекса")
            return {}
        
        if not isinstance(self.object_synonyms, dict):
            logger.error(f"object_synonyms должен быть словарем, получен: {type(self.object_synonyms)}")
            return {}
        
        for object_type, type_synonyms in self.object_synonyms.items():
            if not isinstance(type_synonyms, dict):
                logger.warning(f"type_synonyms для типа '{object_type}' должен быть словарем, получен: {type(type_synonyms)}. Пропускаем.")
                continue
                
            for main_name, synonyms in type_synonyms.items():
                if not isinstance(synonyms, list):
                    logger.warning(f"synonyms для '{main_name}' должен быть списком, получен: {type(synonyms)}. Пропускаем.")
                    continue
                    
                normalized_main = main_name.lower()
                if normalized_main not in self.reverse_object_synonyms:
                    self.reverse_object_synonyms[normalized_main] = []
                
                existing_entry = next((item for item in self.reverse_object_synonyms[normalized_main] 
                                    if item["main_form"] == main_name and item["type"] == object_type), None)
                if not existing_entry:
                    self.reverse_object_synonyms[normalized_main].append({
                        "main_form": main_name,
                        "type": object_type
                    })
                
                for synonym in synonyms:
                    normalized_synonym = synonym.lower()
                    if normalized_synonym not in self.reverse_object_synonyms:
                        self.reverse_object_synonyms[normalized_synonym] = []
                    
                    existing_synonym_entry = next((item for item in self.reverse_object_synonyms[normalized_synonym] 
                                                if item["main_form"] == main_name and item["type"] == object_type), None)
                    if not existing_synonym_entry:
                        self.reverse_object_synonyms[normalized_synonym].append({
                            "main_form": main_name,
                            "type": object_type
                        })
        return self.reverse_object_synonyms
                        
    def resolve_object_synonym(self, object_name: str, object_type: str = "all") -> Dict[str, Any]:
        """
        Разрешает синонимы объектов и возвращает основное название и тип
        """
        if not object_name:
            return {"error": "Название объекта не указано"}
        
        try:
            if not hasattr(self, 'reverse_object_synonyms') or not self.reverse_object_synonyms:
                logger.warning("Индекс синонимов объектов не построен")
                return {
                    "main_form": object_name,
                    "object_type": object_type,
                    "original_name": object_name,
                    "resolved": False
                }
            
            normalized_name = object_name.lower()
            
            if normalized_name in self.reverse_object_synonyms:
                matches = self.reverse_object_synonyms[normalized_name]
                
                if object_type != "all":
                    filtered_matches = [m for m in matches if m["type"] == object_type]
                    if filtered_matches:
                        return {
                            "main_form": filtered_matches[0]["main_form"],
                            "object_type": filtered_matches[0]["type"],
                            "original_name": object_name,
                            "resolved": True
                        }
                
                if matches:
                    return {
                        "main_form": matches[0]["main_form"],
                        "object_type": matches[0]["type"],
                        "original_name": object_name,
                        "resolved": True
                    }
            
            if object_type != "all":
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
    
    def get_synonyms_for_name(self, name: str) -> Dict[str, Any]:
        """
        Возвращает все синонимы для заданного названия вида
        Args:
            name: Название вида (может быть любым синонимом)
        Returns:
            Словарь с основной формой и всеми синонимами
        """
        if not name:
            return {"error": "Название не указано"}
        
        normalized_name = name.lower()
        
        if normalized_name in self.reverse_object_synonyms:
            records = self.reverse_object_synonyms[normalized_name]
            bio_records = [r for r in records if r["type"] == "biological_entity"]
            
            result = {}
            for record in bio_records:
                main_form = record["main_form"]
                synonyms = self.object_synonyms.get("biological_entity", {}).get(main_form, [])
                result[main_form] = synonyms
            
            if result:
                return result
        
        if normalized_name in self.object_synonyms.get("biological_entity", {}):
            main_form = normalized_name
            synonyms = self.object_synonyms["biological_entity"][main_form]
            return {main_form: synonyms}
        
        return {"error": f"Название '{name}' не найдено в базе синонимов"}
    
    def get_object_descriptions(self, object_name: str, object_type: str = "all", in_stoplist: str = "1") -> List[str]:
        """Получает все текстовые описания по названию объекта любого типа с учетом in_stoplist"""
        try:
            all_descriptions = []
            
            search_types = []
            if object_type == "all":
                search_types = ["biological_entity", "geographical_entity", "modern_human_made","organization","research_project","volunteer_initiative","ancient_human_made"]
            else:
                search_types = [object_type]
            
            for entity_type in search_types:
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
        object_name: Optional[str] = None
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
                object_name=object_name
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
            
            search_types = []
            if object_type == "all":
                search_types = ["biological_entity", "geographical_entity", "modern_human_made","organization","research_project","volunteer_initiative","ancient_human_made"]
            else:
                search_types = [object_type]
            
            for entity_type in search_types:
                descriptions = self.relational_service.get_object_descriptions_with_embedding(
                    object_name, entity_type, query_embedding, limit, similarity_threshold, in_stoplist
                )
                if descriptions:
                    all_descriptions.extend(descriptions)
            
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
            
            search_types = []
            if object_type == "all":
                search_types = ["biological_entity", "geographical_entity", "modern_human_made", 
                            "organization", "research_project", "volunteer_initiative", "ancient_human_made"]
            else:
                search_types = [object_type]
            
            for entity_type in search_types:
                descriptions = self.relational_service.search_objects_by_embedding_only(
                    query_embedding=query_embedding,
                    object_type=entity_type,
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                    in_stoplist=in_stoplist
                )
                if descriptions:
                    all_descriptions.extend(descriptions)
            
            all_descriptions.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            return all_descriptions[:limit]
                
        except Exception as e:
            logger.error(f"Ошибка семантического поиска объектов: {str(e)}")
            return []
    
    def _get_llm(self):
        """Получает LLM сервис"""
        if self.llm_service is None:
            from infrastructure.llm_integration import get_gigachat
            self.llm_service = get_gigachat()
        return self.llm_service
    
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
                "- Начинай ответ с прямого ответа на запрос пользователя, отвечай ТОЛЬКО на него"
                "- При запросе 'Какие другие достопримечательности есть?' нужно описать месторождения из твоей базы и другие достопримечательности которые ты знаешь!\n"
                "- Даже при неполной информации предоставь доступные детали\n\n"
                f"Твоя база знаний:\n{context}\n\n"
                f"Вопрос: {question}\n\n"
                "Ответ:"
            ))
        ])
        
        try:
            chain = prompt | llm
            
            response = chain.invoke({"question": question, "context": context})
            
            logger.debug(f"Полный ответ GigaChat: {response}")
            logger.debug(f"Тип ответа: {type(response)}")
            
            finish_reason = None
            
            if hasattr(response, 'response_metadata'):
                logger.debug(f"response_metadata: {response.response_metadata}")
                finish_reason = response.response_metadata.get('finish_reason')
            
            if not finish_reason and hasattr(response, 'llm_output'):
                logger.debug(f"llm_output: {response.llm_output}")
                if isinstance(response.llm_output, dict):
                    finish_reason = response.llm_output.get('finish_reason')
            
            if not finish_reason and hasattr(response, 'finish_reason'):
                finish_reason = response.finish_reason
                
            if not finish_reason and hasattr(response, 'additional_kwargs'):
                logger.debug(f"additional_kwargs: {response.additional_kwargs}")
                finish_reason = response.additional_kwargs.get('finish_reason')
            
            logger.debug(f"Найден finish_reason: {finish_reason}")
            
            # ФИКС: Преобразуем все numpy.float32 в обычные float
            def convert_floats(obj):
                if isinstance(obj, dict):
                    return {k: convert_floats(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_floats(item) for item in obj]
                elif hasattr(obj, 'dtype') and 'float32' in str(obj.dtype):
                    return float(obj)
                return obj
            
            result_data = {
                "content": response.content.strip() if hasattr(response, 'content') else "",
                "finish_reason": finish_reason,
                "success": finish_reason != 'blacklist'
            }
            
            # Преобразуем все float32 в float
            result_data = convert_floats(result_data)
            
            return result_data
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа GigaChat: {str(e)}")
            
            logger.debug(f"Тип исключения: {type(e)}")
            if hasattr(e, 'response'):
                logger.debug(f"Response в исключении: {e.response}")
            
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
        """
        try:
            resolved_synonym = self.resolve_object_synonym(species_name, "biological_entity")
            
            logger.info(f"🔍 Разрешение синонима для '{species_name}': {resolved_synonym}")
            
            main_name = resolved_synonym.get("main_form", species_name)
            object_type = resolved_synonym.get("object_type", "biological_entity")
            
            synonyms_data = self.get_synonyms_for_name(main_name)
            
            if "error" in synonyms_data:
                synonyms_data = {main_name: []}
            
            logger.info(f"✅ Основное название: '{main_name}', синонимы: {synonyms_data}")
            
            return self.relational_service.search_images_by_features(
                species_name=main_name,
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
                
            relevant_indices = []
            raw_indices = response.get("relevant_descriptions", [])
            
            logger.debug(f"Raw indices from LLM: {raw_indices}, type: {type(raw_indices)}")
            
            if isinstance(raw_indices, (int, str)):
                raw_indices = [raw_indices]
                
            for idx in raw_indices:
                try:
                    if isinstance(idx, str):
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
                            try:
                                num_idx = int(idx)
                                if 0 <= num_idx < len(descriptions):
                                    relevant_indices.append(num_idx)
                            except ValueError:
                                continue
                    elif isinstance(idx, int):
                        if 0 <= idx < len(descriptions):
                            relevant_indices.append(idx)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid index '{idx}': {e}")
                    continue

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
        search_around: bool = False,
        buffer_radius_km: float = 10.0
    ) -> Dict[str, Any]:
        """
        Поиск объектов в заданной области с фильтрацией по типу и имени
        """
        try:
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
            
            inside_count = len([obj for obj in results if obj.get('location_type') == 'inside'])
            around_count = len([obj for obj in results if obj.get('location_type') == 'around'])
            
            if object_name:
                message = f"Найден объект '{object_name}'"
            else:
                type_msg = f"типа '{object_type}'" if object_type else "всех типов"
                subtype_msg = f" (подтип: {object_subtype})" if object_subtype else ""
                message = f"Найдено {len(results)} объектов {type_msg}{subtype_msg}"
            
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
        """
        try:
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
        object_subtype: str = None,
        limit: int = 70
    ) -> Dict[str, Any]:
        """Поиск объектов внутри полигона и в буферной зоне с поддержкой подтипов"""
        try:
            try:
                original_shape = shape(polygon_geojson)
                simplified_shape = original_shape.simplify(0.001, preserve_topology=True)
                optimized_polygon_geojson = mapping(simplified_shape)
                
                orig_coords = str(polygon_geojson)[:100]
                simp_coords = str(optimized_polygon_geojson)[:100]
                logger.info(f"📐 Полигон оптимизирован для поиска.")
            except Exception as e:
                logger.warning(f"Не удалось упростить полигон, используем оригинал: {e}")
                optimized_polygon_geojson = polygon_geojson

            results = self.geo_service.get_objects_in_polygon(
                polygon_geojson=optimized_polygon_geojson,
                buffer_radius_km=buffer_radius_km,
                object_type=object_type,
                object_subtype=object_subtype,
                limit=limit
            )
            
            if buffer_radius_km > 0:
                try:
                    buffer_geometry = self.geo_service.create_buffer_geometry(
                        optimized_polygon_geojson, 
                        buffer_radius_km
                    )
                    
                    if buffer_geometry:
                        logger.info(f"🔪 Обрезка полигонов по буферной зоне {buffer_radius_km} км")
                        results = self.geo_service.clip_geometries_to_buffer(
                            results, 
                            buffer_geometry
                        )
                        logger.info(f"✅ Полигоны обрезаны. Осталось объектов: {len(results)}")
                except Exception as e:
                    logger.error(f"Ошибка обрезки полигонов: {str(e)}")
            
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
                    "location_name": obj.get("geo_name") or obj["name"],
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
            
            if object_subtype:
                answer = f"Найдено {total_count} объектов типа '{object_type}' подтипа '{object_subtype}' в области {area_desc} ({type_summary}). Биологические объекты: {biological_objects_str}"
            else:
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
        in_stoplist: int = 1
    ) -> Dict[str, Any]:
        try:
            start = time.perf_counter()
            results = self.geo_service.get_nearby_objects(
                latitude=latitude,
                longitude=longitude,
                radius_km=radius_km,
                limit=limit,
                object_type=object_type,
                species_name=species_name,
                in_stoplist=in_stoplist
            )
            logger.info(f"Nearby objects search took: {time.perf_counter() - start:.2f}s")
            
            if not results:
                return {
                    "answer": f"В радиусе {radius_km} км не найдено объектов",
                    "objects": []
                }
            
            formatted_results = []
            type_counts = {}
            for obj in results:
                obj_type = obj.get("type", "unknown")
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                formatted_obj = {
                    "name": obj["name"],
                    "distance": f"{obj['distance_km']:.1f} км",
                    "type": obj_type,
                    "geojson": obj["geojson"]
                }
                if obj.get("description"):
                    formatted_obj["description"] = obj["description"][:200] + "..." if len(obj["description"]) > 200 else obj["description"]
                
                formatted_results.append(formatted_obj)
            
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