import json

import os

import logging

from pathlib import Path

import re

from langchain_gigachat import GigaChat

from langchain_core.prompts import ChatPromptTemplate

import psycopg2

from psycopg2.extras import RealDictCursor

from dotenv import load_dotenv

from typing import Any, List, Dict, Optional

from infrastructure.llm_integration import get_gigachat



logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

)

logger = logging.getLogger(__name__)



load_dotenv()



class RelationalService:

    def __init__(self,

                species_synonyms_path: Optional[str] = None

    ):

        self.llm = get_gigachat()

        self.db_config = {

            "dbname": os.getenv("DB_NAME", "eco"),

            "user": os.getenv("DB_USER", "postgres"),

            "password": os.getenv("DB_PASSWORD", "Fdf78yh0a4b!"),

            "host": os.getenv("DB_HOST", "localhost"),

            "port": os.getenv("DB_PORT", "5432")

        }

        self.species_synonyms = self._load_species_synonyms(species_synonyms_path)



    def _validate_sql(self, sql: str) -> bool:

        """Проверяет базовую валидность SQL-запроса"""

        sql = sql.strip().upper()

        valid_starts = ["SELECT", "WITH"]

        return any(sql.startswith(cmd) for cmd in valid_starts) and "FROM" in sql



    def _extract_sql_query(self, llm_response: str) -> str:

        """Извлекает чистый SQL-запрос из ответа LLM"""

        clean_response = re.sub(r'```sql|```|<s>|</s>', '', llm_response).strip()

        semicolon_index = clean_response.find(';')

        if semicolon_index != -1:

            clean_response = clean_response[:semicolon_index + 1]

        return re.sub(r'--.*$', '', clean_response, flags=re.MULTILINE).strip()



    def generate_sql_query(self, user_question: str) -> str:

        """Генерирует SQL-запрос с помощью GigaChat"""

        system_prompt = """

Ты эксперт по SQL и базе данных объектов флоры и фауны байкальской территории. 

Сгенерируй ПОЛНЫЙ SQL-запрос для PostgreSQL, заканчивающийся точкой с запятой. 

Этот запрос без изменений будет сразу использован для запроса в базу данных.

Не добавляй пояснений, комментариев и форматирования (никаких ```sql и <s>).



Схема базы данных:

Основные таблицы контента:

1. image_content (изображения):

   - id, title, description, feature_data (JSONB)

2. text_content (тексты):

   - id, title, content, description, feature_data (JSONB)

3. map_content (карты):

   - id, title, description, geometry (GEOGRAPHY), feature_data (JSONB)

4. biological_entity (биологические виды):

   - id, common_name_ru, scientific_name, description, status, feature_data (JSONB)

5. geographical_entity (геообъекты):

   - id, name_ru, description, feature_data (JSONB)



Связи:

1. entity_geo (связь сущностей с географическими объектами):

   - entity_id, entity_type, geographical_entity_id

2. entity_relation (различные связи между сущностями):

   - source_id, source_type, target_id, target_type, relation_type

3. entity_identifier_link (связь сущностей с идентификаторами):

   - entity_id, entity_type, identifier_id

4. entity_identifier (идентификаторы):

   - id, url, file_path, name_ru, name_latin

   

Правила составления запросов:

1. Для поиска мест произрастания/обитания используй entity_geo:

   - JOIN entity_geo ON entity_geo.entity_id = <id сущности> AND entity_geo.entity_type = '<тип сущности>'

   - JOIN geographical_entity ON geographical_entity.id = entity_geo.geographical_entity_id



2. Для поиска связанных сущностей (например, фото видов) используй entity_relation:

   - JOIN entity_relation ON (source_id = <id1> AND source_type = '<тип1>' AND target_id = <id2> AND target_type = '<тип2>') OR (source_id = <id2> AND source_type = '<тип2>' AND target_id = <id1> AND target_type = '<тип1>')

   - Часто используемые типы связей: 'photo', 'description', 'habitat'



3. Для работы с JSON-полями (feature_data):

   - feature_data->>'attribute_name' для текстовых значений

   - feature_data->'nested'->>'attribute' для вложенных JSON



4. Географические функции:

   - ST_AsGeoJSON(geometry)::json для преобразования геометрии

   - ST_Distance(geography1, geography2) для расчетов расстояний



5. Никогда не добавляй после SQL-запроса дополнительные объяснения, теги или примеры



Примеры правильных запросов:



1. Запрос изображений:

Вопрос: "Покажи изображения даурского ежа"

SQL:

SELECT 

  ei.file_path AS image_path,

  ic.title AS title,

  ic.description AS description

FROM biological_entity be

JOIN entity_relation er ON be.id = er.target_id 

  AND er.target_type = 'biological_entity'

JOIN image_content ic ON ic.id = er.source_id 

  AND er.source_type = 'image_content'

  AND er.relation_type = 'изображение объекта'

JOIN entity_identifier_link eil ON eil.entity_id = ic.id 

  AND eil.entity_type = 'image_content'

JOIN entity_identifier ei ON ei.id = eil.identifier_id

WHERE be.common_name_ru ILIKE '%даурский%еж%';



2. Запрос текстового описания:

Вопрос: "Расскажи о байкальской нерпе"

SQL:

SELECT 

  tc.content

FROM biological_entity be

JOIN entity_relation er ON be.id = er.target_id 

  AND er.target_type = 'biological_entity'

  AND er.relation_type = 'описание объекта'

JOIN text_content tc ON tc.id = er.source_id 

  AND er.source_type = 'text_content'

 WHERE be.common_name_ru ILIKE '%байкальская%нерпа%';

    

3. Запрос конкретного изображения с уточнением:

Вопрос: "Покажи шишку сибирской сосны на ветке"

SQL:

SELECT 

  ei.file_path AS image_path,

  ic.title AS title,

  ic.description AS description

FROM biological_entity be

JOIN entity_relation er ON be.id = er.target_id 

  AND er.target_type = 'biological_entity'

JOIN image_content ic ON ic.id = er.source_id 

  AND er.source_type = 'image_content'

  AND er.relation_type = 'изображение объекта'

JOIN entity_identifier_link eil ON eil.entity_id = ic.id 

  AND eil.entity_type = 'image_content'

JOIN entity_identifier ei ON ei.id = eil.identifier_id

WHERE be.common_name_ru ILIKE '%Шишка%лиственницы%сибирской%'

  AND ic.feature_data->>'flora_type'='Шишка';

     

4. Запрос карты полигонов и точек:

Вопрос: "Покажи где растет Копеечник Зундукский"

SQL:

SELECT 

  ge.name_ru AS location_name,

  CASE 

    WHEN mc.geometry IS NOT NULL THEN ST_AsGeoJSON(mc.geometry)::json

    ELSE NULL

  END AS geojson

FROM biological_entity be

JOIN entity_geo eg ON be.id = eg.entity_id 

  AND eg.entity_type = 'biological_entity'

JOIN geographical_entity ge ON ge.id = eg.geographical_entity_id

LEFT JOIN entity_relation er_geo ON ge.id = er_geo.target_id 

  AND er_geo.target_type = 'geographical_entity'

  AND er_geo.relation_type = 'географическая привязка'

LEFT JOIN map_content mc ON mc.id = er_geo.source_id 

  AND er_geo.source_type = 'map_content'

WHERE be.common_name_ru ILIKE '%копеечник%зундукский%';

   

5. Запрос ближайших мест

Вопрос: "Где можно встретить Ольхонскую полевку рядом с поселком Култук"

SQL:

WITH kultuk_point AS (

  SELECT ST_SetSRID(ST_MakePoint(104.3, 51.6), 4326)::geography AS geom

),

species_locations AS (

  SELECT 

    ge.id AS geo_id,

    ge.name_ru AS location_name,

    mc.geometry AS location_geom

  FROM biological_entity be

  JOIN entity_geo eg ON be.id = eg.entity_id 

    AND eg.entity_type = 'biological_entity'

  JOIN geographical_entity ge ON ge.id = eg.geographical_entity_id

  LEFT JOIN entity_relation er ON ge.id = er.target_id 

    AND er.target_type = 'geographical_entity'

    AND er.relation_type = 'точное местоположение'

  LEFT JOIN map_content mc ON mc.id = er.source_id 

    AND er.source_type = 'map_content'

  WHERE be.common_name_ru ILIKE '%ольхонская%полевка%'

)

SELECT 

  sl.location_name,

  CASE 

    WHEN sl.location_geom IS NOT NULL THEN ST_AsGeoJSON(sl.location_geom)::json

    ELSE NULL

  END AS geojson,

  CASE

    WHEN sl.location_geom IS NOT NULL THEN ST_Distance(sl.location_geom, kp.geom) / 1000

    ELSE NULL

  END AS distance_km

FROM species_locations sl, kultuk_point kp

WHERE 

  (sl.location_geom is NOT NULL and ST_DWithin(sl.location_geom, kp.geom, 10000))

ORDER BY 

  CASE

    WHEN sl.location_geom IS NOT NULL THEN ST_Distance(sl.location_geom, kp.geom)

    ELSE 999999 -- Большое значение для сортировки объектов без геометрии в конец

  END

LIMIT 10;



"""

        

        prompt = ChatPromptTemplate.from_messages([

            ("system", system_prompt),

            ("human", user_question)

        ])

        

        

        chain = prompt | self.llm

        response = chain.invoke({"input": user_question})

        

        

        sql = self._extract_sql_query(response.content)

        

        

        #if not self._validate_sql(sql):

            #sql = self._fix_incomplete_sql(sql)

            

        return sql



          

    def execute_query_with_params(self, sql_query: str, params: tuple = None) -> List[Dict]:

      """Выполняет SQL-запрос в PostgreSQL с поддержкой параметров"""

      conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)

      try:

          with conn.cursor() as cursor:

              if params:

                  cursor.execute(sql_query, params)

              else:

                  cursor.execute(sql_query)

              return cursor.fetchall()

      finally:

          conn.close()



    def process_question(self, question: str) -> Dict:

      try:

          sql_query = self.generate_sql_query(question)

          logger.info(f"Сгенерированный SQL: {sql_query}")

          

          if not self._validate_sql(sql_query):

              logger.error("Невалидный SQL-запрос")

              return {

                  "success": False,

                  "error": "Invalid SQL query",

                  "sql": sql_query

              }

          

          results = self.execute_query(sql_query)

          logger.info(f"Результаты SQL: {results}")

          

          if not results:

              return {

                  "success": True,

                  "answer": "По вашему запросу ничего не найдено",

                  "results": []

              }

          

          formatted_results = []

          for row in results:

              if 'geojson' in row:

                  formatted_results.append({

                      "type": "geo",

                      "geojson": row['geojson'],

                      "name": row.get('common_name', 'Без названия')

                  })

              elif 'image_path' in row:

                  formatted_results.append({

                      "type": "image",

                      "path": row['image_path'],

                      "description": row.get('description', '')

                  })

              elif 'content' in row:

                  formatted_results.append({

                      "type": "text",

                      "content": row['content']

                  })

          

          return {

              "success": True,

              "results": formatted_results

          }

          

      except Exception as e:

          logger.error(f"Ошибка выполнения SQL: {str(e)}", exc_info=True)

          return {

              "success": False,

              "error": f"Database error: {str(e)}",

              "sql": sql_query

          }

          

    def search_images_by_features(

    self,

    species_name: str,

    features: Dict[str, Any],

    synonyms_data: Optional[Dict[str, Any]] = None 

) -> Dict[str, Any]:

        """

        Поиск изображений по названию вида и признакам

        

        Args:

            species_name: Название биологического вида

            features: Словарь с признаками для фильтрации

            synonyms_data: Данные синонимов (опционально)

            

        Returns:

            Результаты поиска изображений

        """

        try:

            if synonyms_data is None:

                synonyms_data = {"main_form": [species_name]}

            

            species_conditions = []

            params = []

            

            if "error" not in synonyms_data:

                if isinstance(synonyms_data, dict):

                    if "main_form" in synonyms_data:

                        all_names = [synonyms_data["main_form"]] + synonyms_data.get("synonyms", [])

                    else:

                        all_names = []

                        for main_form, synonyms in synonyms_data.items():

                            all_names.extend([main_form] + synonyms)

                elif isinstance(synonyms_data, list):

                    all_names = synonyms_data

                else:

                    all_names = [species_name]

                    

                for name in all_names:

                    species_conditions.append("be.common_name_ru ILIKE %s")

                    params.append(f'%{name}%')

            else:

                species_conditions.append("be.common_name_ru ILIKE %s")

                params.append(f'%{species_name}%')

            

            sql_query = """

            SELECT 

                ei.file_path AS image_path,

                ic.title AS title,

                ic.description AS description,

                ic.feature_data AS features,

                be.common_name_ru AS species_name

            FROM biological_entity be

            JOIN entity_relation er ON be.id = er.target_id 

                AND er.target_type = 'biological_entity'

                AND er.relation_type = 'изображение объекта'

            JOIN image_content ic ON ic.id = er.source_id 

                AND er.source_type = 'image_content'

            JOIN entity_identifier_link eil ON eil.entity_id = ic.id 

                AND eil.entity_type = 'image_content'

            JOIN entity_identifier ei ON ei.id = eil.identifier_id

            WHERE (""" + " OR ".join(species_conditions) + ")"

            

            

            feature_conditions = []

            for key, value in features.items():

                if key in ['date', 'season', 'habitat', 'cloudiness', 'fauna_type', 'flora_type']:

                    feature_conditions.append(f"ic.feature_data->>'{key}' ILIKE %s")

                    params.append(f'%{value}%')

                elif key == 'location':

                    feature_conditions.append(

                        "(ic.feature_data->'location'->>'region' ILIKE %s OR "

                        "ic.feature_data->'location'->>'country' ILIKE %s)"

                    )

                    params.extend([f'%{value}%', f'%{value}%'])

                elif key == 'flowering':

                    feature_conditions.append("ic.feature_data->'flower_and_fruit_info'->>'flowering' ILIKE %s")

                    params.append(f'%{value}%')

                elif key == 'fruits_present':

                    # Специальная обработка для "нет"

                    if value.lower() == "нет":

                        # Ищем записи, где fruits_present отсутствует, пустой, или содержит "нет"

                        feature_conditions.append(

                            "(ic.feature_data->'flower_and_fruit_info'->>'fruits_present' IS NULL OR "

                            "ic.feature_data->'flower_and_fruit_info'->>'fruits_present' = '' OR "

                            "ic.feature_data->'flower_and_fruit_info'->>'fruits_present' ILIKE %s)"

                        )

                        params.append('%нет%')

                    else:

                        # Обычный поиск по значению

                        feature_conditions.append("ic.feature_data->'flower_and_fruit_info'->>'fruits_present' ILIKE %s")

                        params.append(f'%{value}%')

                elif key == 'author':

                    feature_conditions.append("ic.feature_data->>'author_photo' ILIKE %s")

                    params.append(f'%{value}%')

            

            if feature_conditions:

                sql_query += " AND " + " AND ".join(feature_conditions)

            

            sql_query += " ORDER BY ic.id LIMIT 50;"

            

            logger.info(f"Searching for species: {species_name}")

            logger.info(f"Using synonyms: {synonyms_data}")

            

            results = self.execute_query(sql_query, tuple(params))

            

            if not results:

                return {

                    "status": "not_found",

                    "message": f"Изображения для '{species_name}' с указанными признаками не найдены",

                    "images": [],

                    "synonyms_used": synonyms_data

                }

            

            images = []

            for row in results:

                image_data = {

                    "image_path": row['image_path'],

                    "title": row['title'],

                    "description": row['description'],

                    "species_name": row['species_name'],

                    "features": row['features'] if row['features'] else {}

                }

                images.append(image_data)

            

            return {

                "status": "success",

                "count": len(images),

                "species": species_name,

                "requested_features": features,

                "synonyms_used": synonyms_data,

                "images": images

            }

            

        except Exception as e:

            logger.error(f"Ошибка поиска изображений по признакам: {str(e)}")

            return {

                "status": "error",

                "message": f"Ошибка при поиске изображений: {str(e)}"

            }

    def search_images_by_features_only(self, features: Dict[str, Any]) -> Dict[str, Any]:

        """

        Поиск изображений только по признакам (без привязки к виду)

        

        Args:

            features: Словарь с признаками для фильтрации

            

        Returns:

            Результаты поиска изображений

        """

        try:

            sql_query = """

            SELECT 

                ei.file_path AS image_path,

                ic.title AS title,

                ic.description AS description,

                ic.feature_data AS features,

                be.common_name_ru AS species_name

            FROM image_content ic

            JOIN entity_identifier_link eil ON eil.entity_id = ic.id 

                AND eil.entity_type = 'image_content'

            JOIN entity_identifier ei ON ei.id = eil.identifier_id

            LEFT JOIN entity_relation er ON ic.id = er.source_id 

                AND er.source_type = 'image_content'

                AND er.relation_type = 'изображение объекта'

            LEFT JOIN biological_entity be ON be.id = er.target_id 

                AND er.target_type = 'biological_entity'

            WHERE 1=1

            """

            

            params = []

            feature_conditions = []

            

            for key, value in features.items():

                if key in ['date', 'season', 'habitat', 'cloudiness', 'fauna_type', 'flora_type']:

                    feature_conditions.append(f"ic.feature_data->>'{key}' ILIKE %s")

                    params.append(f'%{value}%')

                elif key == 'location':

                    feature_conditions.append(

                        "(ic.feature_data->'location'->>'region' ILIKE %s OR "

                        "ic.feature_data->'location'->>'country' ILIKE %s)"

                    )

                    params.extend([f'%{value}%', f'%{value}%'])

                elif key == 'flowering':

                    feature_conditions.append("ic.feature_data->'flower_and_fruit_info'->>'flowering' ILIKE %s")

                    params.append(f'%{value}%')

                elif key == 'fruits_present':

                    feature_conditions.append("ic.feature_data->'flower_and_fruit_info'->>'fruits_present' ILIKE %s")

                    params.append(f'%{value}%')

                elif key == 'author':

                    feature_conditions.append("ic.feature_data->>'author_photo' ILIKE %s")

                    params.append(f'%{value}%')

            

            if feature_conditions:

                sql_query += " AND " + " AND ".join(feature_conditions)

            

            sql_query += " ORDER BY ic.id LIMIT 50;"

            

            logger.info(f"Searching images by features only: {features}")

            logger.info(f"SQL: {sql_query}")

            

            results = self.execute_query(sql_query, tuple(params))

            

            if not results:

                return {

                    "status": "not_found",

                    "message": f"Изображения с указанными признаками не найдены",

                    "images": []

                }

            

            images = []

            for row in results:

                image_data = {

                    "image_path": row['image_path'],

                    "title": row['title'],

                    "description": row['description'],

                    "species_name": row['species_name'],

                    "features": row['features'] if row['features'] else {}

                }

                images.append(image_data)

            

            return {

                "status": "success",

                "count": len(images),

                "requested_features": features,

                "images": images

            }

            

        except Exception as e:

            logger.error(f"Ошибка поиска изображений только по признакам: {str(e)}")

            return {

                "status": "error",

                "message": f"Ошибка при поиске изображений: {str(e)}"

            }

   

                 

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

            

    def get_text_descriptions(self, species_name: str) -> List[str]:

        """Получает все текстовые описания по названию вида"""

        query = """

        SELECT tc.content, tc.structured_data

        FROM biological_entity be

        JOIN entity_relation er ON be.id = er.target_id 

            AND er.target_type = 'biological_entity'

            AND er.relation_type = 'описание объекта'

        JOIN text_content tc ON tc.id = er.source_id 

            AND er.source_type = 'text_content'

        WHERE be.common_name_ru ILIKE %s;

        """

        try:

            results = self.execute_query(query, (f'%{species_name}%',))

            descriptions = []

            

            for row in results:

                content = row['content']

                structured_data = row.get('structured_data')

                

                if not content and structured_data:

                    extracted_content = self._extract_content_from_structured_data(structured_data)

                    if extracted_content:

                        descriptions.append(extracted_content)

                elif content:

                    descriptions.append(content)

                    

            return descriptions

            

        except Exception as e:

            logger.error(f"Ошибка получения описаний для '{species_name}': {str(e)}")

            return []

        

    def get_object_descriptions(self, object_name: str, object_type: str) -> List[str]:

        """Получает текстовые описания для объектов любого типа"""

        query = """

        SELECT tc.content, tc.structured_data

        FROM {table_name} be

        JOIN entity_relation er ON be.id = er.target_id 

            AND er.target_type = %(object_type)s

            AND er.relation_type = 'описание объекта'

        JOIN text_content tc ON tc.id = er.source_id 

            AND er.source_type = 'text_content'

        WHERE {name_field} ILIKE %(object_name)s;

        """

        

        try:

            # Определяем таблицу и поле имени в зависимости от типа объекта

            table_map = {

                "biological_entity": {"table": "biological_entity", "name_field": "be.common_name_ru"},

                "geographical_entity": {"table": "geographical_entity", "name_field": "be.name_ru"}, 

                "modern_human_made": {"table": "modern_human_made", "name_field": "be.name_ru"},

                "ancient_human_made": {"table": "ancient_human_made", "name_field": "be.name_ru"},

                "organization": {"table": "organization", "name_field": "be.name_ru"},

                "research_project": {"table": "research_project", "name_field": "be.title"},

                "volunteer_initiative": {"table": "volunteer_initiative", "name_field": "be.name_ru"},

            }

            

            if object_type not in table_map:

                return []

                

            table_info = table_map[object_type]

            formatted_query = query.format(

                table_name=table_info["table"], 

                name_field=table_info["name_field"]

            )

            

            results = self.execute_query(

                formatted_query, 

                {'object_type': object_type, 'object_name': f'%{object_name}%'}

            )

            

            descriptions = []

            for row in results:

                content = row['content']

                structured_data = row.get('structured_data')

                

                if not content and structured_data:

                    extracted_content = self._extract_content_from_structured_data(structured_data)

                    if extracted_content:

                        descriptions.append(extracted_content)

                elif content:

                    descriptions.append(content)

                    

            return descriptions

            

        except Exception as e:

            logger.error(f"Ошибка получения описаний для '{object_name}': {str(e)}")

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

            # Определяем типы объектов для поиска

            search_types = []

            if object_type == "all":

                # search_types = ["biological_entity", "geographical_entity", "modern_human_made", 

                #             "organization", "research_project", "volunteer_initiative", "ancient_human_made"]

                search_types = ["geographical_entity"]

            else:

                search_types = [object_type]

            

            all_descriptions = []

            

            for entity_type in search_types:

                descriptions = self._get_descriptions_by_filters_for_type(

                    filter_data=filter_data,

                    object_type=entity_type,

                    limit=limit

                )

                if descriptions:

                    all_descriptions.extend(descriptions)

            

            return all_descriptions[:limit]

                

        except Exception as e:

            logger.error(f"Ошибка поиска объектов по фильтрам: {str(e)}")

            return []

    def _get_descriptions_by_filters_for_type(

        self,

        filter_data: Dict[str, Any],

        object_type: str,

        limit: int

    ) -> List[Dict]:

        """

        Поиск описаний для конкретного типа объекта по фильтрам

        """

        query = """

        SELECT 

            tc.content, 

            tc.structured_data,

            be.name_ru as object_name

        FROM {table_name} be

        JOIN entity_relation er ON be.id = er.target_id 

            AND er.target_type = %(object_type)s

            AND er.relation_type = 'описание объекта'

        JOIN text_content tc ON tc.id = er.source_id 

            AND er.source_type = 'text_content'

        WHERE 1=1

        """

        

        params = {

            'object_type': object_type,

            'limit': limit

        }

        

        # Добавляем условия фильтрации

        conditions = []

        

        # Фильтрация по location_info

        if 'location_info' in filter_data:

            location_info = filter_data['location_info']

            

            if 'exact_location' in location_info:
                if location_info['exact_location']!=None:
                    exact_location = location_info['exact_location']

                    conditions.append(

                        "(tc.structured_data->'geographical_info'->>'name' ILIKE %(exact_location)s OR "

                        "tc.structured_data->'geographical_info'->>'description' ILIKE %(exact_location)s)"

                    )

                    params['exact_location'] = f'%{exact_location}%'

            

            if 'region' in location_info:

                region = location_info['region']

                conditions.append(

                    "tc.structured_data->'geographical_info'->>'name' ILIKE %(region)s"

                )

                params['region'] = f'%{region}%'

        

        # Фильтрация по geo_type

        if 'geo_type' in filter_data:

            geo_type = filter_data['geo_type']

            

            type_conditions = []

            

            if 'primary_type' in geo_type and geo_type['primary_type']:

                primary_types = geo_type['primary_type']

                if isinstance(primary_types, list):

                    for i, primary_type in enumerate(primary_types):

                        param_name = f'primary_type_{i}'

                        type_conditions.append(f"tc.structured_data->'geographical_info'->>'object_type' ILIKE %({param_name})s")

                        params[param_name] = f'%{primary_type}%'

            

            if 'specific_types' in geo_type and geo_type['specific_types']:

                specific_types = geo_type['specific_types']

                if isinstance(specific_types, list):

                    for i, specific_type in enumerate(specific_types):

                        param_name = f'specific_type_{i}'

                        type_conditions.append(f"tc.structured_data->'geographical_info'->>'object_type' ILIKE %({param_name})s")

                        params[param_name] = f'%{specific_type}%'

            

            if type_conditions:

                conditions.append("(" + " OR ".join(type_conditions) + ")")

        

        # Определяем таблицу и поле имени в зависимости от типа объекта

        table_map = {

            "biological_entity": {"table": "biological_entity", "name_field": "be.common_name_ru"},

            "geographical_entity": {"table": "geographical_entity", "name_field": "be.name_ru"},

            "modern_human_made": {"table": "modern_human_made", "name_field": "be.name_ru"},

            "ancient_human_made": {"table": "ancient_human_made", "name_field": "be.name_ru"},

            "organization": {"table": "organization", "name_field": "be.name_ru"},

            "research_project": {"table": "research_project", "name_field": "be.title"},

            "volunteer_initiative": {"table": "volunteer_initiative", "name_field": "be.name_ru"},

        }

        

        if object_type not in table_map:

            return []

            

        table_info = table_map[object_type]

        formatted_query = query.format(table_name=table_info["table"])

        

        # Добавляем условия фильтрации в запрос

        if conditions:

            formatted_query += " AND " + " AND ".join(conditions)

        

        formatted_query += " LIMIT %(limit)s;"

        

        logger.debug(f"Выполняется поиск по фильтрам для типа: '{object_type}'")

        logger.debug(f"SQL запрос: {formatted_query}")

        logger.debug(f"Параметры: {params}")

        

        try:

            results = self.execute_query(formatted_query, params)

            

            formatted_results = []

            for row in results:

                content = row.get('content')

                structured_data = row.get('structured_data')

                object_name = row.get('object_name')

                

                final_content = content

                if not final_content and structured_data:

                    final_content = self._extract_content_from_structured_data(structured_data)

                

                if final_content:

                    formatted_results.append({

                        "content": final_content,

                        "source": "structured_data" if not content and structured_data else "content",

                        "object_name": object_name,

                        "object_type": object_type

                    })

            

            return formatted_results

            

        except Exception as e:

            logger.error(f"Ошибка выполнения запроса по фильтрам для '{object_type}': {str(e)}")

            return []

        

    def search_objects_by_embedding_only(

    self, 

    query_embedding: List[float],

    object_type: str,

    limit: int = 10,

    similarity_threshold: float = 0.05

) -> List[Dict]:

        """

        Поиск объектов только по эмбеддингу запроса (без указания конкретного имени объекта)

        """

        query = """

        SELECT 

            tc.content, 

            tc.structured_data, 

            1 - (tc.embedding <=> %(embedding)s::vector) as similarity,

            be.common_name_ru as object_name,

            'biological_entity' as object_type

        FROM text_content tc

        JOIN entity_relation er ON tc.id = er.source_id 

            AND er.source_type = 'text_content'

            AND er.relation_type = 'описание объекта'

        JOIN biological_entity be ON be.id = er.target_id 

            AND er.target_type = 'biological_entity'

        WHERE 1 - (tc.embedding <=> %(embedding)s::vector) > %(similarity_threshold)s

        ORDER BY similarity DESC

        LIMIT %(limit)s;

        """

        

        try:

            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            

            params = {

                'embedding': embedding_str,

                'similarity_threshold': similarity_threshold,

                'limit': limit

            }

            

            results = self.execute_query(query, params)

            

            if not results:

                return []

            

            formatted_results = []

            for row in results:

                try:

                    content = row.get('content')

                    structured_data = row.get('structured_data')

                    similarity = row.get('similarity')

                    object_name = row.get('object_name')

                    

                    final_content = content

                    if not final_content and structured_data:

                        final_content = self._extract_content_from_structured_data(structured_data)

                    

                    if final_content and similarity is not None:

                        formatted_results.append({

                            "content": final_content,

                            "similarity": float(similarity),

                            "source": "structured_data" if not content and structured_data else "content",

                            "object_name": object_name,

                            "object_type": row.get('object_type', 'unknown')

                        })

                except Exception as e:

                    logger.error(f"Ошибка обработки строки результата: {str(e)}")

                    continue

            

            return formatted_results

            

        except Exception as e:

            logger.error(f"Ошибка семантического поиска объектов: {str(e)}")

            return []

        

    def get_object_descriptions_with_embedding(self, object_name: str, object_type: str, 

                                            query_embedding: List[float],

                                            limit: int = 10, 

                                            similarity_threshold: float = 0.1) -> List[Dict]:

        """

        УНИВЕРСАЛЬНАЯ функция для получения текстовых описаний объектов 

        с учетом схожести эмбеддингов (ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ).

        """

        

        query = """

        SELECT 

            tc.content, 

            tc.structured_data, 

            1 - (tc.embedding <=> %(embedding)s::vector) as similarity

        FROM {table_name} be

        JOIN entity_relation er ON be.id = er.target_id 

            AND er.target_type = %(object_type)s

            AND er.relation_type = 'описание объекта'

        JOIN text_content tc ON tc.id = er.source_id 

            AND er.source_type = 'text_content'

        WHERE {name_field} ILIKE %(object_name)s

          AND 1 - (tc.embedding <=> %(embedding)s::vector) > %(similarity_threshold)s

        ORDER BY similarity DESC

        LIMIT %(limit)s;

        """

        

        try:

            # <-- ВАЖНО: Добавляем лог, чтобы видеть, для какого типа объекта мы работаем

            logger.debug(f"Выполняется векторный поиск для типа: '{object_type}' с именем: '{object_name}'")



            table_map = {

                "biological_entity": {"table": "biological_entity", "name_field": "be.common_name_ru"},

                "geographical_entity": {"table": "geographical_entity", "name_field": "be.name_ru"},

                "modern_human_made": {"table": "modern_human_made", "name_field": "be.name_ru"},

                "ancient_human_made": {"table": "ancient_human_made", "name_field": "be.name_ru"},

                "organization": {"table": "organization", "name_field": "be.name_ru"},

                "research_project": {"table": "research_project", "name_field": "be.title"},

                "volunteer_initiative": {"table": "volunteer_initiative", "name_field": "be.name_ru"}

            }

            

            if object_type not in table_map:

                logger.warning(f"Неизвестный тип объекта для поиска: {object_type}")

                return []

                

            table_info = table_map[object_type]

            

            # Используем именованные параметры для надежности, как и в функции без эмбеддингов

            formatted_query = query.format(

                table_name=table_info["table"], 

                name_field=table_info["name_field"]

            )

            

            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            

            params = {

                'embedding': embedding_str,

                'object_type': object_type,

                'object_name': f'%{object_name}%',

                'similarity_threshold': similarity_threshold,

                'limit': limit

            }

            

            # <-- ВАЖНО: Логируем сам запрос и параметры перед выполнением

            logger.debug(f"Сформированный SQL-запрос:\n{formatted_query}")

            logger.debug(f"Параметры запроса: {params}")



            results = self.execute_query(formatted_query, params)

            

            if not results:

                logger.debug(f"Для типа '{object_type}' не найдено результатов в БД.")

                return []

            

            logger.debug(f"Найдено {len(results)} результатов для типа '{object_type}'.")

            

            formatted_results = []

            for row in results:

                try:

                    content = row.get('content')

                    structured_data = row.get('structured_data')

                    similarity = row.get('similarity')

                    

                    final_content = content

                    if not final_content and structured_data:

                        final_content = self._extract_content_from_structured_data(structured_data)

                    

                    if final_content and similarity is not None:

                        formatted_results.append({

                            "content": final_content,

                            "similarity": float(similarity),

                            "source": "structured_data" if not content and structured_data else "content"

                        })

                except Exception as e:

                    logger.error(f"Ошибка обработки строки результата: {str(e)}")

                    continue

            

            return formatted_results

            

        except Exception as e:

            logger.error(f"Критическая ошибка в get_object_descriptions_with_embedding для '{object_name}': {str(e)}")

            return []



# УДАЛИТЕ СТАРУЮ ФУНКЦИЮ get_text_descriptions_with_embedding, ОНА БОЛЬШЕ НЕ НУЖНА

# def get_text_descriptions_with_embedding(...):

#     ...

        

    def get_text_descriptions_with_embedding(self, species_name: str, query_embedding: List[float], 

                               limit: int = 10, similarity_threshold: float = 0.5) -> List[Dict]:

        """Получает текстовые описания с учетом схожести эмбеддингов"""

        query = """

        SELECT tc.content, tc.structured_data, 1 - (tc.embedding <=> %s::vector) as similarity

        FROM biological_entity be

        JOIN entity_relation er ON be.id = er.target_id 

            AND er.target_type = 'biological_entity'

            AND er.relation_type = 'описание объекта'

        JOIN text_content tc ON tc.id = er.source_id 

            AND er.source_type = 'text_content'

        WHERE be.common_name_ru ILIKE %s

        AND 1 - (tc.embedding <=> %s::vector) > %s

        ORDER BY similarity DESC

        LIMIT %s;

        """

        try:

            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            

            results = self.execute_query(

                query, 

                (embedding_str, f'%{species_name}%', embedding_str, similarity_threshold, limit)

            )

            

            if not results:

                return []

            

            formatted_results = []

            for row in results:

                try:

                    content = row.get('content')

                    structured_data = row.get('structured_data')

                    similarity = row.get('similarity')

                    

                    # Если content пустой, извлекаем из structured_data

                    final_content = content

                    if not final_content and structured_data:

                        final_content = self._extract_content_from_structured_data(structured_data)

                    

                    if final_content and similarity is not None:

                        formatted_results.append({

                            "content": final_content,

                            "similarity": float(similarity),

                            "source": "structured_data" if not content and structured_data else "content"

                        })

                        

                except Exception as e:

                    logger.error(f"Error processing row: {str(e)}")

                    continue

            

            return formatted_results

            

        except Exception as e:

            logger.error(f"Ошибка получения описаний с эмбеддингом для '{species_name}': {str(e)}", exc_info=True)

            return []
        
    def search_objects_by_name(
    self,
    object_name: str,
    object_type: Optional[str] = None,
    object_subtype: Optional[str] = None,
    limit: int = 20
) -> List[Dict]:
        """
        Поиск объектов по имени с возможной фильтрацией по типу и подтипу
        """
        query = """
        SELECT 
            ge.id,
            ge.name_ru AS name,
            ge.description,
            ge.feature_data,
            'geographical_entity' AS type,
            ST_AsGeoJSON(mc.geometry)::json AS geojson,
            ST_GeometryType(mc.geometry) AS geometry_type
        FROM geographical_entity ge
        JOIN entity_geo eg ON ge.id = eg.geographical_entity_id
        JOIN map_content mc ON eg.entity_id = mc.id AND eg.entity_type = 'map_content'
        WHERE ge.name_ru ILIKE %(object_name)s
        """
        
        params = {
            'object_name': f'%{object_name}%',
            'limit': limit
        }
        
        conditions = []
        
        # Фильтрация по типу объекта
        if object_type:
            conditions.append("""
                (
                    ge.feature_data->'geo_type'->'primary_type' ? %(object_type)s
                    OR ge.feature_data->'geo_type'->'specific_types' ? %(object_type)s
                    OR ge.feature_data->>'information_type' = %(object_type)s
                )
            """)
            params['object_type'] = object_type
        
        # Фильтрация по подтипу
        if object_subtype:
            conditions.append("ge.feature_data->'geo_type'->'specific_types' ? %(object_subtype)s")
            params['object_subtype'] = object_subtype
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " ORDER BY ge.name_ru LIMIT %(limit)s;"
        
        try:
            results = self.execute_query(query, params)
            
            formatted_results = []
            for row in results:
                features = row['feature_data'] or {}
                geo_type = features.get('geo_type', {})
                
                formatted_results.append({
                    "id": row['id'],
                    "name": row['name'],
                    "description": row['description'],
                    "type": row['type'],
                    "geometry_type": row['geometry_type'],
                    "geojson": row['geojson'],
                    "features": features,
                    "primary_types": geo_type.get('primary_type', []),
                    "specific_types": geo_type.get('specific_types', [])
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Ошибка поиска объектов по имени '{object_name}': {str(e)}")
            return []
        
    def get_objects_in_area_by_type(
    self,
    area_geometry: dict,
    object_type: Optional[str] = None,
    object_subtype: Optional[str] = None,
    object_name: Optional[str] = None,
    limit: int = 20
) -> List[Dict]:
        """
        Поиск географических объектов в заданной области с фильтрацией
        """
        area_geojson_str = json.dumps(area_geometry)
        
        query = """
        WITH search_area AS (
            SELECT ST_GeomFromGeoJSON(%(area_geojson)s)::geography AS geom
        )
        SELECT
            ge.id,
            ge.name_ru AS name,
            ge.description,
            ge.feature_data,
            'geographical_entity' AS type,
            ST_AsGeoJSON(mc.geometry)::json AS geojson,
            ST_GeometryType(mc.geometry) AS geometry_type
        FROM geographical_entity ge
        JOIN entity_geo eg ON ge.id = eg.geographical_entity_id
        JOIN map_content mc ON eg.entity_id = mc.id AND eg.entity_type = 'map_content'
        CROSS JOIN search_area sa
        WHERE ST_Intersects(mc.geometry, sa.geom)
        """
        
        params = {
            'area_geojson': area_geojson_str,
            'limit': limit
        }
        
        conditions = []
        
        # Фильтрация по имени объекта
        if object_name:
            conditions.append("ge.name_ru ILIKE %(object_name)s")
            params['object_name'] = f'%{object_name}%'
        
        # Фильтрация по типу объекта
        if object_type:
            conditions.append("""
                (
                    ge.feature_data->'geo_type'->'primary_type' ? %(object_type)s
                    OR ge.feature_data->'geo_type'->'specific_types' ? %(object_type)s
                    OR ge.feature_data->>'information_type' = %(object_type)s
                )
            """)
            params['object_type'] = object_type
        
        # Фильтрация по подтипу
        if object_subtype:
            conditions.append("ge.feature_data->'geo_type'->'specific_types' ? %(object_subtype)s")
            params['object_subtype'] = object_subtype
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " ORDER BY ge.name_ru LIMIT %(limit)s;"
        
        try:
            results = self.execute_query(query, params)
            
            formatted_results = []
            for row in results:
                features = row['feature_data'] or {}
                geo_type = features.get('geo_type', {})
                
                formatted_results.append({
                    "id": row['id'],
                    "name": row['name'],
                    "description": row['description'],
                    "type": row['type'],
                    "geometry_type": row['geometry_type'],
                    "geojson": row['geojson'],
                    "features": features,
                    "primary_types": geo_type.get('primary_type', []),
                    "specific_types": geo_type.get('specific_types', [])
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Ошибка поиска объектов по типу в области: {str(e)}")
            return []
            
    def find_area_geometry(self, area_name: str) -> Optional[Dict]:
        """
        Поиск полигона области в таблице map_content
        """
        query = """
        SELECT 
            mc.id,
            mc.title,
            ST_AsGeoJSON(mc.geometry)::json AS geometry_geojson,  -- Преобразуем WKB в GeoJSON
            mc.feature_data,
            'map_content' as source
        FROM map_content mc
        WHERE mc.title ILIKE %s 
        AND (
            mc.feature_data->>'type' IN ('geographical_entity', 'region', 'city', 'area', 'polygon')
            OR ST_GeometryType(mc.geometry) != 'ST_Point'
        )
        ORDER BY 
            CASE 
                WHEN mc.title ILIKE %s THEN 0
                WHEN mc.feature_data->>'type' IN ('city', 'region') THEN 1
                ELSE 2
            END,
            LENGTH(mc.title)
        LIMIT 1
        """
        
        try:
            results = self.execute_query(query, (f'%{area_name}%', area_name))
            
            if results:
                row = results[0]
                geometry_geojson = row['geometry_geojson']
                
                logger.debug(f"Найдена геометрия для '{area_name}': {geometry_geojson.get('type') if geometry_geojson else 'None'}")
                
                if geometry_geojson:
                    return {
                        "geometry": geometry_geojson,
                        "area_info": {
                            "id": row['id'],
                            "title": row['title'],
                            "source": row['source'],
                            "feature_data": row['feature_data']
                        }
                    }
        
            # Если не нашли в map_content, пробуем через географические сущности
            geo_query = """
            SELECT 
                ge.id,
                ge.name_ru as title,
                ST_AsGeoJSON(mc.geometry)::json AS geometry_geojson,  -- Преобразуем WKB в GeoJSON
                mc.feature_data,
                'geographical_entity' as source
            FROM geographical_entity ge
            JOIN entity_geo eg ON ge.id = eg.geographical_entity_id
            JOIN map_content mc ON eg.entity_id = mc.id AND eg.entity_type = 'map_content'
            WHERE ge.name_ru ILIKE %s
            AND ST_GeometryType(mc.geometry) != 'ST_Point'
            ORDER BY 
                CASE 
                    WHEN ge.name_ru ILIKE %s THEN 0
                    ELSE 1
                END,
                LENGTH(ge.name_ru)
            LIMIT 1
            """
            
            geo_results = self.execute_query(geo_query, (f'%{area_name}%', area_name))
            
            if geo_results:
                row = geo_results[0]
                geometry_geojson = row['geometry_geojson']
                
                logger.debug(f"Найдена геометрия (geo) для '{area_name}': {geometry_geojson.get('type') if geometry_geojson else 'None'}")
                
                if geometry_geojson:
                    return {
                        "geometry": geometry_geojson,
                        "area_info": {
                            "id": row['id'],
                            "title": row['title'],
                            "source": row['source'],
                            "feature_data": row['feature_data']
                        }
                    }
            
            logger.warning(f"Полигон для области '{area_name}' не найден")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка поиска полигона области '{area_name}': {str(e)}")
            return None
        
    def _extract_content_from_structured_data(self, structured_data: Dict) -> str:

        """

        Извлекает и форматирует текстовый контент из structured_data

        """

        if not structured_data:

            return ""

        

        content_sections = []

        

        # Русские названия разделов (заменяем английские ключи)

        section_titles = {

            'morphology': 'Морфология',

            'ecology': 'Экология', 

            'distribution': 'Распространение',

            'phenology': 'Фенология',

            'significance': 'Значение',

            'conservation': 'Охранный статус',

            'taxonomy': 'Таксономия'

        }

        

        # Русские названия полей (заменяем английские ключи)

        field_titles = {

            'general_description': 'Общее описание',

            'habitat': 'Местообитание',

            'ecological_role': 'Экологическая роль',

            'geographical_range': 'Географический ареал',

            'baikal_region_status': 'Статус в Байкальском регионе',

            'flowering_period': 'Период цветения',

            'fruiting_period': 'Период плодоношения',

            'practical_use': 'Практическое использование',

            'scientific_value': 'Научное значение',

            'soil_preferences': 'Предпочтения к почве',

            'light_requirements': 'Требования к свету',

            'species_interactions': 'Взаимодействие с другими видами',

            'moisture_requirements': 'Требования к влаге',

            'genus': 'Род',

            'family': 'Семейство', 

            'species': 'Вид',

            'vegetation_period': 'Период вегетации',

            'stem': 'Стебель',

            'roots': 'Корни',

            'fruits': 'Плоды',

            'leaves': 'Листья',

            'flowers': 'Цветы',

            'threats': 'Угрозы',

            'red_book_status': 'Статус в Красной книге',

            'protection_status': 'Статус охраны',

            'protected_areas': 'Охраняемые территории'

        }

        

        for section, section_data in structured_data.items():

            if section not in section_titles or not isinstance(section_data, dict):

                continue

                

            section_content = []

            for field, value in section_data.items():

                if (value not in ['-', '', None] and 

                    isinstance(value, str) and 

                    len(value.strip()) > 0):

                    

                    # Используем русское название поля, если доступно

                    field_title = field_titles.get(field, field)

                    section_content.append(f"{field_title}: {value}")

            

            if section_content:

                content_sections.append(

                    f"{section_titles[section]}:\n" + 

                    "\n".join(f"• {line}" for line in section_content)

                )

        

        return "\n\n".join(content_sections) if content_sections else ""

            

    def execute_query(self, sql_query: str, params: tuple = None) -> List[Dict]:

        """Выполняет SQL-запрос в PostgreSQL с поддержкой параметров"""

        conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)

        try:

            with conn.cursor() as cursor:

                logger.debug(f"Executing SQL: {sql_query}")

                logger.debug(f"With params: {params}")

                

                if params:

                    cursor.execute(sql_query, params)

                else:

                    cursor.execute(sql_query)

                    

                results = cursor.fetchall()

                logger.debug(f"Raw results from DB: {results}")

                return results

        except Exception as e:

            logger.error(f"Database error: {str(e)}", exc_info=True)

            return []

        finally:

            conn.close()