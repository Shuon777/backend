import os

import logging

from pathlib import Path

from typing import Dict, List, Any, Optional,Tuple,Union

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import HumanMessage,SystemMessage

from langchain_core.output_parsers import JsonOutputParser

from langchain_core.documents import Document

from core import query_parser, response_formatter

from infrastructure.vector_stores import init_vector_stores

from core.relational_service import RelationalService

import json

from .sql_agent import SQLAgent

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

    faiss_index_path: str,

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

        self.faiss_index_path = faiss_index_path

        self.embedding_model_path = embedding_model_path

        self.query_modifier = query_parser.QueryModifier()

        self.llm_service = llm_service or get_gigachat()

        self.relational_service = RelationalService(species_synonyms_path)

        self.sql_agent = SQLAgent() 

        self.sql_tools = self.sql_agent.get_tools()

        self.geo_service = GeoService()

        self.vectorstores = init_vector_stores(

            faiss_index_path=faiss_index_path,

            embedding_model_path=embedding_model_path

        )

        self.species_synonyms = self._load_species_synonyms(species_synonyms_path)

        self._build_reverse_synonyms_index()



        self.embedding_model = HuggingFaceEmbeddings(

            model_name=embedding_model_path,

            model_kwargs={'device': 'cpu'},

            encode_kwargs={'normalize_embeddings': False}

        )

    def _init_gigachat(self):

        if self.llm is None:

            self.llm = get_gigachat()

            

    def _get_llm(self):

        if self.llm_service is None:

            self.llm_service = get_gigachat()

        return self.llm_service

    

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

    

    def get_object_descriptions(self, object_name: str, object_type: str = "all") -> List[str]:

        """Получает все текстовые описания по названию объекта любого типа"""

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

                descriptions = self.relational_service.get_object_descriptions(object_name, entity_type)

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

        limit: int = 10

    ) -> List[Dict]:

        """

        Поиск описаний объектов по фильтрам из JSON body

        

        Args:

            filter_data: Словарь с фильтрами

            object_type: Тип объекта для поиска

            limit: Максимальное количество результатов

            

        Returns:

            Список описаний объектов

        """

        try:

            return self.relational_service.get_object_descriptions_by_filters(

                filter_data=filter_data,

                object_type=object_type,

                limit=limit

            )

                

        except Exception as e:

            logger.error(f"Ошибка поиска объектов по фильтрам: {str(e)}")

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
            if any(phrase in error_str for phrase in ['blacklist', 'restricted', 'content policy', 'finish_reason']):
                return {
                    "content": "К сожалению, я не могу сгенерировать ответ на этот вопрос на основе доступной информации.",
                    "finish_reason": "blacklist",
                    "success": False
                }
            
            return {
                "content": "Извините, не удалось сгенерировать ответ на основе доступной информации.",
                "finish_reason": "error",
                "success": False
            }

    

    def search_objects_by_embedding(

    self, 

    query_embedding: List[float],

    object_type: str = "all",

    limit: int = 10,

    similarity_threshold: float = 0.05

) -> List[Dict]:

        """

        Поиск объектов по семантическому сходству с запросом

        

        Args:

            query_embedding: Эмбеддинг запроса

            object_type: Тип объекта для поиска

            limit: Максимальное количество результатов

            similarity_threshold: Порог схожести

            

        Returns:

            Список описаний объектов с информацией о схожести

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

                # Для этого нужно передать пустой object_name и полагаться на эмбеддинг

                descriptions = self.relational_service.search_objects_by_embedding_only(

                    query_embedding=query_embedding,

                    object_type=entity_type,

                    limit=limit,

                    similarity_threshold=similarity_threshold

                )

                if descriptions:

                    all_descriptions.extend(descriptions)

            

            # Сортируем по схожести и ограничиваем количество

            all_descriptions.sort(key=lambda x: x.get("similarity", 0), reverse=True)

            return all_descriptions[:limit]

                

        except Exception as e:

            logger.error(f"Ошибка семантического поиска объектов: {str(e)}")

            return []

        

    def get_object_descriptions_with_embedding(self, object_name: str, object_type: str, 

                                        query_embedding: List[float], 

                                        limit: int = 10, 

                                        similarity_threshold: float = 0.05) -> List[Dict]:

        """Получает текстовые описания объектов с учетом схожести эмбеддингов"""

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

                    object_name, entity_type, query_embedding, limit, similarity_threshold

                )

                if descriptions:

                    all_descriptions.extend(descriptions)

            

            # Сортируем по схожести и ограничиваем количество

            all_descriptions.sort(key=lambda x: x.get("similarity", 0), reverse=True)

            return all_descriptions[:limit]

                

        except Exception as e:

            logger.error(f"Ошибка получения описаний с эмбеддингом для '{object_name}': {str(e)}")

            return []

    

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

            

    def get_text_descriptions(self, species_name: str) -> List[str]:

        """Получает все текстовые описания по названию вида с использованием синонимов"""

        try:

            synonyms_data = self.get_synonyms_for_name(species_name)

            all_descriptions = []

            

            if "error" not in synonyms_data:

                for main_form, synonyms in synonyms_data.items():

                    all_names = [main_form] + synonyms

                    for name in all_names:

                        descriptions = self.relational_service.get_text_descriptions(name)

                        if descriptions:

                            all_descriptions.extend(descriptions)

            else:

                descriptions = self.relational_service.get_text_descriptions(species_name)

                if descriptions:

                    all_descriptions.extend(descriptions)

                    

            return list(set(all_descriptions))

                    

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

            

    def _filter_docs_with_gigachat(self, query: str, docs: List[Document]) -> List[int]:

        """Фильтрация документов через LLM"""

        llm = self._get_llm()

        

        if not docs or not query:

            return list(range(len(docs))) if docs else []



        docs_text = "\n\n".join(

            f"Документ {i}:\n{doc.page_content[:1000]}"

            for i, doc in enumerate(docs))

        

        prompt = ChatPromptTemplate.from_messages([

        SystemMessage(content=(

            "Ты эксперт по Байкальской экосистеме. Фильтруй документы СТРОГО по критериям:\n"

            "1. Документ ДОЛЖЕН содержать ВСЕ элементы запроса: [вид растения] + [тип местности] + [погода]\n"

            "2. Частичное соответствие = отклонение\n"

            "3. Если отсутствует ЛЮБОЙ элемент → документ нерелевантен\n\n"

            

            "## Формат ответа ТОЛЬКО JSON:\n"

            "{\n"

            "  \"relevant_docs\": [список int-индексов] | [],\n"

            "  \"no_relevant_docs\": bool\n"

            "}\n\n"

            

            "## Критические примеры:\n"

            "Запрос: 'Покажи Остролодочник в ясную погоду на побережье'\n"

            "→ Допустимый ответ: {\"relevant_docs\": [], \"no_relevant_docs\": true} (если нет полного соответствия)\n\n"

            

            "Запрос: 'Черепоплодник щетинистоватый на песчаном пляже'\n"

            "→ Недопустимо: указание документа про чаек\n\n"

            

            "## Инструкции:\n"

            "- Не интерпретируй запрос расширительно\n"

            "- Растения: точное соответствие названий\n"

            "- Погода: только явные указания в тексте\n"

            "- ЛЮБОЕ нарушение → no_relevant_docs: true"

        )),

        HumanMessage(content=(

            f"ЗАПРОС: {query}\n\n"

            f"ДОКУМЕНТЫ ДЛЯ АНАЛИЗА:\n{docs_text}\n\n"

            "ВЕРНИ JSON ОТВЕТ БЕЗ КОММЕНТАРИЕВ:"

        ))

    ])



        try:

            chain = prompt | llm | JsonOutputParser()

            response = chain.invoke({"query": query, "docs": docs_text})

            

            # Валидация ответа

            if response.get("no_relevant_docs", False):

                return []

                

            relevant_indices = [

                idx for idx in response.get("relevant_docs", [])

                if isinstance(idx, int) and 0 <= idx < len(docs)

            ]

            return relevant_indices

            

        except Exception as e:

            logger.error(f"Ошибка фильтрации через GigaChat: {str(e)}")

            return list(range(len(docs)))





    def ask_question(

    self,

    question: str,

    similarity_threshold: float = 0.52,

    similarity_deviation: float = None,

    use_gigachat: bool = False,

    user_id: Optional[str] = None,

    debug_mode: bool = False,

    knowledge_base_type: str = "vector",

    query_formatter: Optional[str] = None,

    strict_filter: bool = False,

    filter_words: Optional[List[str]] = None  # Новый параметр

) -> Dict[str, Any]:

        """Основной метод для обработки вопросов"""

        try:

            logger.debug(f"Начало обработки вопроса: '{question}' (knowledge_base={knowledge_base_type}, agent={query_formatter == 'agent'}, strict_filter={strict_filter})")

            if knowledge_base_type == "relational":

                if query_formatter == "agent":

                    return self._ask_agent(

                        question=question,

                        debug_mode=debug_mode

                    )

                else:

                    return self._ask_relational(

                        question=question,

                        query_formatter=query_formatter,

                        debug_mode=debug_mode,

                    )

            else:

                return self._ask_vector(

            question=question,

            similarity_threshold=similarity_threshold,

            similarity_deviation=similarity_deviation,

            use_gigachat=use_gigachat,

            debug_mode=debug_mode,

            user_id=user_id,

            strict_filter=strict_filter,

            filter_words=filter_words  

        )

        except Exception as e:

            logger.error(f"Критическая ошибка в ask_question: {str(e)}")

            return self._build_error_response(str(e), debug_mode)

        

    def _ask_vector(

        self,

        question: str,

        similarity_threshold: float,

        similarity_deviation: Optional[float],

        use_gigachat: bool,

        debug_mode: bool,

        user_id: Optional[str] = None,

        strict_filter: bool = False,

        filter_words: Optional[List[str]] = None  

    ) -> Dict[str, Any]:

        """Обработка запросов к векторной базе знаний

        

        Args:

            question: Вопрос пользователя

            similarity_threshold: Минимальный порог схожести

            similarity_deviation: Максимальное отклонение от лучшего результата

            use_gigachat: Использовать GigaChat для фильтрации

            debug_mode: Включить отладочную информацию

            

        Returns:

            Форматированный ответ с результатами поиска

        """

        modified_query = self.query_modifier.modify(question)

        requested_types = self._extract_requested_types(modified_query)

        

        search_stores = self._get_search_stores(requested_types)

        if not search_stores:

            return self._build_empty_response(

                "Нет подходящего векторного хранилища",

                debug_mode

            )

        

        all_results, all_scores = self._search_documents(

            question,

            search_stores,

            similarity_threshold,

            debug_mode

        )

        

        if not all_results:

            return self._build_empty_response(

                "По вашему запросу ничего не найдено",

                debug_mode

            )

        

        if similarity_deviation is not None:

            filtered_results, filtered_scores = self._apply_dynamic_threshold(

                all_results,

                all_scores,

                similarity_deviation,

                debug_mode

            )

        else:

            filtered_results = all_results

            filtered_scores = all_scores

        

        if use_gigachat:

            try:

                logger.debug(f"Фильтрация GigaChat. До: {len(filtered_results)} документов")

                filtered_indices = self._filter_docs_with_gigachat(question, filtered_results)

                logger.debug(f"После фильтрации: {len(filtered_indices)} документов")

                if not filtered_indices:

                    logger.info("GigaChat отфильтровал ВСЕ документы как нерелевантные")

                    return self._build_empty_response(

                        "По вашему запросу ничего не найдено",

                        debug_mode

                    )

                    

                final_results = [filtered_results[i] for i in filtered_indices]

                final_scores = [filtered_scores[i] for i in filtered_indices] if debug_mode else None

                logger.debug(f"Gigachat результаты: {final_results}")

                logger.debug(f"Gigachat оценки векторной БД: {final_scores}")

            except Exception as e:

                logger.error(f"Ошибка при фильтрации через GigaChat: {str(e)}")

                final_results = filtered_results

                final_scores = filtered_scores if debug_mode else None

        else:

            final_results = filtered_results

            final_scores = filtered_scores if debug_mode else None

        if strict_filter and filter_words:

            logger.debug(f"Applying strict filter with words: {filter_words}")

            temp_results = []

            

            for doc in final_results:

                content = doc.page_content.lower()

                all_words_found = True

                

                for word in filter_words:

                    word_lower = word.lower()

                    synonyms = self.species_synonyms.get(word_lower, [word_lower])

                    word_found = any(synonym.lower() in content for synonym in synonyms)

                    

                    if not word_found:

                        all_words_found = False

                        break

                        

                if all_words_found:

                    temp_results.append(doc)

                    

            final_results = temp_results

            if debug_mode and final_scores is not None:

                final_scores = [s for i, s in enumerate(final_scores) 

                            if i < len(final_results)]

        logger.debug(final_results)

        if not final_results:

            return self._build_empty_response(

                "По вашему запросу ничего не найдено",

                debug_mode

            )

        return self._format_response(

            final_results,

            requested_types,

            final_scores,

            debug_mode,

            user_id=user_id

        )

    

    def _ask_relational(

        self,

        question: str,

        query_formatter: str,

        debug_mode: bool

    ) -> Dict[str, Any]:

        """Обработка запросов к реляционной базе знаний

        

        Args:

            question: Вопрос пользователя

            query_formatter: Способ формирования запроса ("gigachat")

            debug_mode: Включить отладочную информацию

            

        Returns:

            Форматированный ответ с результатами поиска

        """

        try:

            if query_formatter == "gigachat":

                result = self.relational_service.process_question(question)

            else:

                raise ValueError(f"Unsupported formatter: {query_formatter}")

            

            if not result["success"]:

                return self._build_empty_response(

                    result.get("error", "Ошибка реляционной БД"),

                    debug_mode

                )

            

            return self._format_relational_response(

                results=result["results"],

                debug_mode=debug_mode

            )

            

        except Exception as e:

            logger.error(f"Ошибка реляционной БЗ: {str(e)}")

            return self._build_error_response(str(e), debug_mode)

        

    def _get_agent_prompt(self) -> ChatPromptTemplate:

        """Промпт для агента"""

        return ChatPromptTemplate.from_messages([

            ("system", (

                "Ты эксперт по Байкальской природной территории. "

                "Используй инструменты для работы с базой данных. "

                "Анализируй запрос пользователя и выбирай подходящий инструмент.\n\n"

                "Доступные инструменты:\n"

                "1. simple_search_tool: Для поиска по названию вида\n"

                "2. attribute_search_tool: Для поиска с дополнительными атрибутами\n"

                "3. geo_search_tool: Для геопоиска\n"

                "4. text_description_tool: Для получения текстового описания\n\n"

                "Примеры использования:\n"

                "- 'Покажи даурского ежа' → simple_search_tool\n"

                "- 'Покажи шишку сибирской сосны' → attribute_search_tool\n"

                "- 'Где можно встретить ольхонскую полевку' → geo_search_tool\n"

                "- 'Расскажи о байкальской нерпе' → text_description_tool\n"

            )),

            ("human", "{input}")

        ])

    def _format_agent_response(self, raw_response: str, debug_mode: bool) -> Dict[str, Any]:

        """Форматирует ответ агента с учетом типа контента"""

        response = {

            "answer": "",

            "multi_url": {

                "image_urls": [],

                "file_urls": [],

                "geo_places": []

            }

        }

        

        try:

            # Сначала парсим ответ агента

            agent_response = json.loads(raw_response)

            

            # Если есть информация об ошибке

            if "error" in agent_response:

                response["answer"] = f"Ошибка: {agent_response['error']}"

                if debug_mode:

                    response["debug_info"] = agent_response.get("debug", {})

                return response

            

            # Извлекаем основной ответ

            answer_data = agent_response.get("answer", "")

            

            try:

                # Пытаемся распарсить как JSON (если это результаты поиска)

                results = json.loads(answer_data)

            except:

                # Если не JSON, значит это простой текст

                results = answer_data

            

            # Обработка результатов

            if isinstance(results, list):

                for item in results:

                    if isinstance(item, dict) and "image_path" in item:

                        image_url = item["image_path"]

                        response["multi_url"]["image_urls"].append(image_url)

                        

                        desc = f"{item.get('name', 'Изображение')}"

                        if item.get('description'):

                            desc += f" - {item['description']}"

                        response["answer"] += f"📷 {desc}\n"

                    

                    elif isinstance(item, dict) and "text_content" in item:

                        response["answer"] += f"📝 {item['text_content']}\n"

                    

                    elif isinstance(item, dict) and "error" in item:

                        response["answer"] += f"⚠️ {item['error']}\n"

                        if "suggestions" in item:

                            response["answer"] += "Рекомендации:\n- " + "\n- ".join(item["suggestions"]) + "\n"

            else:

                # Если ответ не список, просто выводим его

                response["answer"] = str(results)

            

            # Если ответ пустой, но есть параметры

            if not response["answer"] and "parameters" in agent_response:

                params = agent_response["parameters"]

                response["answer"] = (

                    f"Запрос обработан. Параметры: "

                    f"species_name='{params.get('species_name', '')}', "

                    f"search_type='{params.get('search_type', '')}'"

                )

            

        except json.JSONDecodeError:

            response["answer"] = raw_response

        

        if debug_mode:

            response["debug_info"] = agent_response.get("debug", {})

            

        return response

            

    def _ask_agent(self, question: str, debug_mode: bool) -> Dict[str, Any]:

        """Делегирует выполнение агенту"""

        try:

            logger.debug(f"Starting agent with question: {question}")

            response = self.sql_agent.ask(question)

            logger.debug(f"Agent response: {response}")

            return self._format_agent_response(response, debug_mode)

        except Exception as e:

            logger.error(f"Ошибка SQL-агента: {str(e)}", exc_info=True)

            return self._build_error_response(str(e), debug_mode)

        

    def _format_relational_response(

    self,

    results: List[Dict],

    debug_mode: bool

) -> Dict[str, Any]:

        """Форматирует ответ из реляционной базы данных

        

        Args:

            results: Список результатов из БД

            debug_mode: Включить отладочную информацию

            

        Returns:

            Словарь с форматированным ответом

        """

        answer_parts = []

        image_urls = []

        file_urls = []

        geo_places = []



        for item in results:

            if item.get("type") == "text":

                content = item.get("content", "")

                if content:

                    answer_parts.append(content)

            

            elif item.get("type") == "image":

                image_path = item.get("path") or item.get("image_path") or item.get("file_path")

                if image_path:

                    image_urls.append(image_path)

                    description = item.get("description") or item.get("image_caption") or "Изображение"

                    answer_parts.append(f"📷 {description}")

            

            elif item.get("type") == "file":

                file_path = item.get("path") or item.get("file_path")

                if file_path:

                    file_urls.append(file_path)

                    description = item.get("description") or item.get("name") or "Файл"

                    answer_parts.append(f"📄 {description}")

            

            if item.get("location_name"):

                geo_places.append(item["location_name"])

            elif item.get("geojson"):

                geo_places.append("Геообъект")



        answer = "\n\n".join(answer_parts) if answer_parts else "Данные не найдены"

        logger.debug(f"{image_urls} - картинки")

        response = {

            "answer": answer,

            "multi_url": {

                "image_urls": image_urls,

                "file_urls": file_urls,

                "geo_places": geo_places

            }

        }

        logger.debug(f"{response} - итоговый ответ")

        if debug_mode:

            response["debug_info"] = {

                "results_count": len(results),

                "query_type": "relational",

                "raw_results": results[:3]

            }

            

        return response

    

    def _apply_dynamic_threshold(

        self,

        results: List[Document],

        scores: List[float],

        deviation: float,

        debug_mode: bool

    ) -> Tuple[List[Document], List[float]]:

        """Фильтрует результаты по отклонению от максимальной схожести"""

        if not results or not scores:

            return results, scores

            

        max_score = max(scores)

        threshold_dynamic = max_score - deviation

        

        filtered_results = []

        filtered_scores = []

        

        for i, score in enumerate(scores):

            if score >= threshold_dynamic:

                filtered_results.append(results[i])

                filtered_scores.append(score)

        

        if debug_mode:

            logger.debug(

                f"Динамический порог: max={max_score:.4f}, "

                f"deviation={deviation}, "

                f"threshold={threshold_dynamic:.4f}, "

                f"осталось {len(filtered_results)} результатов"

            )

            

        return filtered_results, filtered_scores

    

    def _extract_requested_types(self, modified_query: str) -> Optional[List[str]]:

        """Извлечение запрошенных типов из модифицированного запроса"""

        if 'type:' not in modified_query:

            return None

            

        parts = modified_query.split('type:')

        type_part = parts[1].strip()

        return [t.strip() for t in type_part.split('|')]



    def _get_search_stores(self, requested_types: Optional[List[str]]) -> List[str]:

        """Определение хранилищ для поиска на основе запрошенных типов"""

        if not requested_types:

            return list(self.vectorstores.keys())

            

        return [

            t for t in requested_types

            if t in self.vectorstores

        ]

        

    def _search_documents(self, query, search_stores, similarity_threshold, debug_mode):

        """Поиск документов в указанных хранилищах"""

        all_results = []

        all_scores = []

        

        for store_type in search_stores:

            try:

                vectorstore = self.vectorstores[store_type]

                scored_results = vectorstore.similarity_search_with_score(query, k=10)

                

                for doc, score in scored_results:

                    similarity_value = 1 / (1 + score)

                    if similarity_value >= similarity_threshold:

                        all_results.append(doc)

                        all_scores.append(float(similarity_value))

                        

                logger.debug(f"Найдено {len(scored_results)} результатов в {store_type}")

                

            except Exception as e:

                logger.error(f"Ошибка поиска в хранилище {store_type}: {str(e)}")

                if debug_mode:

                    all_results.append(Document(

                        page_content=f"Ошибка поиска в хранилище {store_type}: {str(e)}",

                        metadata={"type": "Ошибка", "name": "Ошибка поиска"}

                    ))

                    all_scores.append(0.0)

        

        return all_results, all_scores 

    
    def get_objects_in_area_by_type(
    self,
    area_geometry: dict,
    object_type: Optional[str] = None,
    object_subtype: Optional[str] = None,
    object_name: Optional[str] = None,
    limit: int = 20
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
                limit=limit
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
            
            # Формируем ответное сообщение
            if object_name:
                message = f"Найден объект '{object_name}' в указанной области"
            else:
                type_msg = f"типа '{object_type}'" if object_type else "всех типов"
                subtype_msg = f" (подтип: {object_subtype})" if object_subtype else ""
                message = f"Найдено {len(results)} объектов {type_msg}{subtype_msg} в указанной области"
            
            return {
                "answer": message,
                "objects": results,
                "area_geometry": area_geometry
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

    limit: int = 20

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

    object_type: str = None,

    radius_km: float = 10,

    species_name: Optional[Union[str, List[str]]] = None 

) -> Dict[str, Any]:

        try:

            # Исправленный вызов GeoService

            start = time.perf_counter()

            results = self.geo_service.get_nearby_objects(

                latitude, 

                longitude,

                object_type=object_type,

                radius_km=radius_km,

                species_name=species_name 

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

            

    def _format_response(

    self,

    results,

    requested_types,

    scores,

    debug_mode,

    user_id=None, 

    error=None

):

        answer, urls = response_formatter.format_response(

        results,

        requested_types,

        scores,

        debug_mode,

        user_id=user_id 

        )

          

        response = {

            "answer": answer,

            "multi_url": {

                "image_urls": urls.get("images", []),

                "file_urls": urls.get("files", []),

                "geo_places": urls.get("geo_places", [])

            }

        }

        

        if debug_mode:

            response["debug_info"] = {

                "results_count": len(results),

                "error": error

            }

            

        return response

    

    def _build_empty_response(self, message, debug_mode):

        response = {

            "answer": f"Извините, {message.lower()}",

            "multi_url": {

                "image_urls": [],

                "file_urls": [],

                "geo_places": []

            }

        }

        if debug_mode:

            response["debug_info"] = {"message": message}

        return response

    

    def _build_error_response(self, error, debug_mode):

        response = {

            "answer": "Произошла ошибка при обработке запроса",

            "multi_url": {

                "image_urls": [],

                "file_urls": [],

                "geo_places": []

            }

        }

        if debug_mode:

            response["debug_info"] = {"error": error}

        return response
