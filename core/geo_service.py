import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from typing import List, Dict,Optional,Union,Tuple
import json
import time
from functools import lru_cache
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeoService:
    def __init__(self):
        self.db_config = {
            "dbname": os.getenv("DB_NAME", "eco"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "Fdf78yh0a4b!"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }
        self.species_synonyms = {
            "копеечник зундукский": ["Hedysarum zundukii","копеечник зундукский"],
            "астрагал ольхонский": ["Astragalus olchonensis","астрагал ольхонский"],
            "лапчатка ольхонская": ["Potentilla olchonensis","лапчатка ольхонская"],
            "остролодочник": ["Oxytropis","остролодочник"],
            "мак попова": ["Papaver popovii","мак попова"],
            "черепоплодник щетинистоватый": ["Craniospermum subvillosum","черепоплодник щетинистоватый","черепоплодник"],
            "эдельвейс": ["Leontopodium","эдельвейс","эдэльвейс","эдельвэйс"],
            "тимьян": ["тимьян", "thymus", "чабрец", "чабер", "богородская трава", "темьян"],
            "иван чай": ["кипрей","иван чай","иван-чай", "иванчай", "кипрей узколистный", "Chamerion angustifolium", "Epilobium angustifolium"],
            "овсяница ленская": ["Festuca lenensis","овсяница ленская"],
            "кедр сибирский": ["сибирский кедр","кедр сибирский","сосна сибирская кедровая"],
        }
        self._get_nearby_objects_cached.cache_clear()
    def clear_cache(self):
        self._get_nearby_objects_cached.cache_clear()
    def _expand_species_names(self, species_names: Union[str, List[str]]) -> List[str]:
        """Приводит все синонимы к основному названию вида"""
        if isinstance(species_names, str):
            species_names = [species_names]
            
        canonical_names = set()
        
        # Сначала создаем обратный словарь синонимов: синоним -> основное название
        reverse_synonyms = {}
        for canonical, synonyms in self.species_synonyms.items():
            canonical_lower = canonical.lower()
            for syn in synonyms:
                reverse_synonyms[syn.lower()] = canonical_lower
            # Добавляем и само каноническое название в словарь
            reverse_synonyms[canonical_lower] = canonical_lower
        
        for name in species_names:
            name_lower = name.lower()
            # Ищем основное название для каждого введенного имени
            canonical_name = reverse_synonyms.get(name_lower, name_lower)
            canonical_names.add(canonical_name)
        
        return list(canonical_names)
    
    @lru_cache(maxsize=1000)
    def _get_nearby_objects_cached(
            self, 
            lat_key: float, 
            lon_key: float, 
            radius_km: float, 
            limit: int,
            obj_type: Optional[str],
            species_tuple: Optional[Tuple[str]],
            in_stoplist: int  # Добавить параметр
        ) -> List[Dict]:
        """
        Кэшированная версия запроса объектов поблизости
        Использует округленные координаты для ключа кэша
        """
        # Преобразуем кортеж обратно в список
        species_list = list(species_tuple) if species_tuple else None
        if self._get_nearby_objects_cached.cache_info().hits > 0:
            logger.debug(f"Cache hit for {lat_key},{lon_key}")
        else:
            logger.debug(f"Cache miss for {lat_key},{lon_key}")
        
        # Вызываем основной метод
        return self._get_nearby_objects_uncached(
            latitude=lat_key,
            longitude=lon_key,
            radius_km=radius_km,
            limit=limit,
            object_type=obj_type,
            species_name=species_list,
            in_stoplist=in_stoplist  # Передать параметр
        )
        
    def create_buffer_geometry(self, original_geometry: dict, buffer_radius_km: float) -> Optional[dict]:
        """
        Создает буферную геометрию вокруг исходной геометрии используя PostGIS
        """
        conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        try:
            with conn.cursor() as cursor:
                original_geojson_str = json.dumps(original_geometry)
                
                query = """
                SELECT ST_AsGeoJSON(
                    ST_Buffer(
                        ST_GeomFromGeoJSON(%s)::geography,
                        %s * 1000
                    )
                )::json AS buffer_geojson;
                """
                
                cursor.execute(query, (original_geojson_str, buffer_radius_km))
                result = cursor.fetchone()
                
                if result and result['buffer_geojson']:
                    return result['buffer_geojson']
                return None
                
        except Exception as e:
            logger.error(f"Ошибка создания буферной геометрии: {str(e)}")
            return None
        finally:
            conn.close()
        
    def _get_nearby_objects_uncached(
    self, 
    latitude: float, 
    longitude: float, 
    radius_km: float = 10, 
    limit: int = 20,
    object_type: str = None,
    species_name: Optional[Union[str, List[str]]] = None,
    in_stoplist: Union[str, int] = 1  # Принимаем и строку и число
) -> List[Dict]:
        """
        Основная реализация поиска объектов (без кэширования)
        """
        conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        try:
            with conn.cursor() as cursor:
                # ПРЕОБРАЗОВАНИЕ in_stoplist в число с обработкой строковых значений
                try:
                    if isinstance(in_stoplist, str):
                        if in_stoplist.lower() in ['false', 'true']:
                            # Если пришло "false" или "true", используем значение по умолчанию
                            in_stoplist_int = 1
                        else:
                            in_stoplist_int = int(in_stoplist)
                    else:
                        in_stoplist_int = int(in_stoplist)
                except (ValueError, TypeError):
                    # В случае ошибки используем значение по умолчанию
                    in_stoplist_int = 1
                
                # Подготовка параметров запроса
                params = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'radius_km': radius_km,
                    'limit': limit,
                    'in_stoplist': in_stoplist_int  # Используем преобразованное число
                }

                # Обработка синонимов видов
                if species_name:
                    expanded_species = self._expand_species_names(species_name)
                    params['species_patterns'] = [f"%{name}%" for name in expanded_species]

                # Основной запрос с проверкой in_stoplist для биологических видов
                query = """
                WITH user_point AS (
                    SELECT ST_SetSRID(ST_MakePoint(%(longitude)s, %(latitude)s), 4326)::geography AS geom
                )
                SELECT DISTINCT ON (entities.id)
                    entities.id,
                    entities.name,
                    entities.description,
                    entities.type,
                    ST_AsGeoJSON(mc.geometry)::json AS geojson,
                    ST_Distance(mc.geometry, up.geom) / 1000 AS distance_km
                FROM ("""
                
                # Собираем части запроса для разных типов объектов
                entity_parts = []
                
                # Биологические сущности (с учетом синонимов)
                if object_type is None or object_type == "biological_entity":
                    biological_part = """
                        SELECT 
                            be.id,
                            be.common_name_ru AS name,
                            be.description,
                            'biological_entity' AS type,
                            eg.geographical_entity_id
                        FROM biological_entity be
                        JOIN entity_geo eg ON be.id = eg.entity_id 
                            AND eg.entity_type = 'biological_entity'
                    """
                    entity_parts.append(biological_part)
                
                # Географические объекты
                if object_type is None or object_type == "geographical_entity":
                    geographical_part = """
                        SELECT 
                            ge.id,
                            ge.name_ru AS name,
                            ge.description,
                            'geographical_entity' AS type,
                            ge.id AS geographical_entity_id
                        FROM geographical_entity ge
                    """
                    entity_parts.append(geographical_part)
                
                # Комбинируем все части запроса
                entities_query = " UNION ALL ".join(entity_parts)
                
                # Условия для фильтрации по виду (с учетом синонимов И in_stoplist)
                species_join = ""
                species_condition = ""
                if species_name:
                    species_join = """
                        JOIN entity_geo eg_species ON entities.geographical_entity_id = eg_species.geographical_entity_id
                        JOIN biological_entity be_species ON eg_species.entity_id = be_species.id 
                            AND eg_species.entity_type = 'biological_entity'
                    """
                    species_condition = """
                        AND (
                            be_species.common_name_ru ILIKE ANY(%(species_patterns)s) 
                            OR be_species.scientific_name ILIKE ANY(%(species_patterns)s)
                        )
                        -- ФИЛЬТРАЦИЯ ПО STOPLIST: используем переданный параметр с обработкой строковых значений
                        AND (
                            be_species.feature_data->>'in_stoplist' IS NULL 
                            OR be_species.feature_data->>'in_stoplist' = 'false'
                            OR be_species.feature_data->>'in_stoplist' = 'true'
                            OR (be_species.feature_data->>'in_stoplist')::integer <= %(in_stoplist)s
                        )
                    """
                
                # Формируем окончательный запрос
                final_query = query + entities_query + f"""
                ) entities
                {species_join}
                JOIN map_content mc ON mc.id IN (
                    SELECT entity_id 
                    FROM entity_geo 
                    WHERE geographical_entity_id = entities.geographical_entity_id
                    AND entity_type = 'map_content'
                )
                CROSS JOIN user_point up
                WHERE ST_DWithin(mc.geometry, up.geom, %(radius_km)s * 1000)
                {species_condition}
                ORDER BY entities.id, distance_km 
                LIMIT %(limit)s;
                """
                
                logger.debug(f"Executing query with params: {params}")
                debug_query = cursor.mogrify(final_query, params).decode('utf-8')
                logger.debug(f"Full SQL query:\n{debug_query}")
                start_time = time.time()
                cursor.execute(final_query, params)
                execution_time = time.time() - start_time
                logger.debug(f"Query executed in {execution_time:.4f} seconds")
                
                results = cursor.fetchall()
                # Преобразование результатов в более удобный формат
                formatted_results = []
                for row in results:
                    formatted_row = dict(row)
                    # Преобразование GeoJSON в Python-объект, если нужно
                    if isinstance(formatted_row['geojson'], str):
                        formatted_row['geojson'] = json.loads(formatted_row['geojson'])
                    formatted_results.append(formatted_row)
                
                return formatted_results

        except Exception as e:
            logger.error(f"Ошибка поиска объектов: {str(e)}", exc_info=True)
            return []
        finally:
            conn.close()
            
    def get_nearby_objects(
    self, 
    latitude: float, 
    longitude: float, 
    radius_km: float = 10, 
    limit: int = 20,
    object_type: str = None,
    species_name: Optional[Union[str, List[str]]] = None,
    in_stoplist: int = 1  # Новый параметр
) -> List[Dict]:
        """
        Поиск объектов в радиусе от заданной точки с кэшированием
        """
        # Округляем координаты для ключа кэша (4 знака ≈ 10 метров)
        lat_key = round(latitude, 4)
        lon_key = round(longitude, 4)
        
        # Округляем радиус для ключа кэша
        radius_key = round(radius_km, 1)
        
        # Преобразуем species_name в кортеж для хеширования
        species_tuple = None
        if species_name:
            if isinstance(species_name, str):
                species_tuple = (species_name.lower(),)
            else:
                species_tuple = tuple(sorted(name.lower() for name in species_name))
        else:
            species_tuple = None
        
        # Вызываем кэшированную версию с учетом in_stoplist
        return self._get_nearby_objects_cached(
            lat_key=lat_key,
            lon_key=lon_key,
            radius_km=radius_key,
            limit=limit,
            obj_type=object_type,
            species_tuple=species_tuple,
            in_stoplist=in_stoplist  # Передаем уровень стоплиста
        )
            
    def get_objects_in_polygon(
    self,
    polygon_geojson: dict,
    buffer_radius_km: float = 0,
    object_type: str = None,
    object_subtype: str = None,
    limit: int = 20
) -> List[Dict]:
        """Поиск объектов внутри полигона и в буферной зоне вокруг него с поддержкой подтипов"""
        conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        try:
            with conn.cursor() as cursor:
                polygon_str = json.dumps(polygon_geojson)
                
                # Базовый запрос с поддержкой фильтрации по подтипам
                query = """
                WITH area AS (
                    SELECT 
                        ST_Buffer(
                            ST_GeomFromGeoJSON(%(polygon_str)s)::geography,
                            %(buffer_radius_km)s * 1000
                        ) AS geom
                ),
                entities AS (
                    {entities_query}
                )
                SELECT
                    entities.id,
                    entities.name,
                    entities.description,
                    entities.type,
                    entities.features,
                    entities.geo_name,  -- ДОБАВЛЕНО: название географического объекта
                    ST_AsGeoJSON(mc.geometry)::json AS geojson,
                    ST_Distance(
                        CASE 
                            WHEN ST_GeometryType(mc.geometry) = 'ST_Point' 
                            THEN mc.geometry 
                            ELSE ST_Centroid(mc.geometry)
                        END,
                        ST_Centroid(a.geom)
                    ) / 1000 AS distance_km
                FROM entities
                JOIN map_content mc ON mc.id IN (
                    SELECT entity_id 
                    FROM entity_geo 
                    WHERE geographical_entity_id = entities.geographical_entity_id
                    AND entity_type = 'map_content'
                )
                CROSS JOIN area a
                WHERE ST_Intersects(mc.geometry, a.geom)
                {subtype_condition}
                ORDER BY distance_km
                LIMIT %(limit)s;
                """
                
                # Формируем запрос для сущностей с учетом типа
                entities_parts = []
                params = {
                    'polygon_str': polygon_str,
                    'buffer_radius_km': buffer_radius_km,
                    'limit': limit
                }
                
                # Биологические сущности - ИСПРАВЛЕНО: добавляем название географического объекта
                if not object_type or object_type == "biological_entity":
                    biological_part = """
                    SELECT 
                        be.id,
                        be.common_name_ru AS name,  -- Название биологического вида
                        be.description,
                        be.type,
                        be.feature_data AS features,
                        eg.geographical_entity_id,
                        ge.name_ru AS geo_name  -- ДОБАВЛЕНО: название географического объекта
                    FROM biological_entity be
                    JOIN entity_geo eg ON be.id = eg.entity_id 
                        AND eg.entity_type = 'biological_entity'
                    JOIN geographical_entity ge ON eg.geographical_entity_id = ge.id  -- ДОБАВЛЕНО: джойн с географическими объектами
                    """
                    entities_parts.append(biological_part)
                
                # Географические объекты - ИСПРАВЛЕНО: используем name_ru как geo_name
                if not object_type or object_type == "geographical_entity":
                    geographical_part = """
                    SELECT 
                        ge.id,
                        ge.name_ru AS name,
                        ge.description,
                        'geographical_entity' AS type,
                        ge.feature_data AS features,
                        ge.id AS geographical_entity_id,
                        ge.name_ru AS geo_name  -- ДОБАВЛЕНО: название географического объекта
                    FROM geographical_entity ge
                    """
                    entities_parts.append(geographical_part)
                
                # Современные рукотворные объекты - ИСПРАВЛЕНО: добавляем название географического объекта
                if not object_type or object_type == "modern_human_made":
                    modern_part = """
                    SELECT 
                        mhm.id,
                        mhm.name_ru AS name,
                        mhm.description,
                        'modern_human_made' AS type,
                        mhm.feature_data AS features,
                        eg.geographical_entity_id,
                        ge.name_ru AS geo_name  -- ДОБАВЛЕНО: название географического объекта
                    FROM modern_human_made mhm
                    JOIN entity_geo eg ON mhm.id = eg.entity_id 
                        AND eg.entity_type = 'modern_human_made'
                    JOIN geographical_entity ge ON eg.geographical_entity_id = ge.id  -- ДОБАВЛЕНО: джойн с географическими объектами
                    """
                    entities_parts.append(modern_part)
                
                # Древние рукотворные объекты - ИСПРАВЛЕНО: добавляем название географического объекта
                if not object_type or object_type == "ancient_human_made":
                    ancient_part = """
                    SELECT 
                        ahm.id,
                        ahm.name_ru AS name,
                        ahm.description,
                        'ancient_human_made' AS type,
                        ahm.feature_data AS features,
                        eg.geographical_entity_id,
                        ge.name_ru AS geo_name  -- ДОБАВЛЕНО: название географического объекта
                    FROM ancient_human_made ahm
                    JOIN entity_geo eg ON ahm.id = eg.entity_id 
                        AND eg.entity_type = 'ancient_human_made'
                    JOIN geographical_entity ge ON eg.geographical_entity_id = ge.id  -- ДОБАВЛЕНО: джойн с географическими объектами
                    """
                    entities_parts.append(ancient_part)
                
                # Организации - ИСПРАВЛЕНО: добавляем название географического объекта
                if not object_type or object_type == "organization":
                    org_part = """
                    SELECT 
                        org.id,
                        org.name_ru AS name,
                        org.description,
                        'organization' AS type,
                        org.feature_data AS features,
                        eg.geographical_entity_id,
                        ge.name_ru AS geo_name  -- ДОБАВЛЕНО: название географического объекта
                    FROM organization org
                    JOIN entity_geo eg ON org.id = eg.entity_id 
                        AND eg.entity_type = 'organization'
                    JOIN geographical_entity ge ON eg.geographical_entity_id = ge.id  -- ДОБАВЛЕНО: джойн с географическими объектами
                    """
                    entities_parts.append(org_part)
                
                # Исследовательские проекты - ИСПРАВЛЕНО: добавляем название географического объекта
                if not object_type or object_type == "research_project":
                    research_part = """
                    SELECT 
                        rp.id,
                        rp.title AS name,
                        rp.description,
                        'research_project' AS type,
                        rp.feature_data AS features,
                        eg.geographical_entity_id,
                        ge.name_ru AS geo_name  -- ДОБАВЛЕНО: название географического объекта
                    FROM research_project rp
                    JOIN entity_geo eg ON rp.id = eg.entity_id 
                        AND eg.entity_type = 'research_project'
                    JOIN geographical_entity ge ON eg.geographical_entity_id = ge.id  -- ДОБАВЛЕНО: джойн с географическими объектами
                    """
                    entities_parts.append(research_part)
                
                # Волонтерские инициативы - ИСПРАВЛЕНО: добавляем название географического объекта
                if not object_type or object_type == "volunteer_initiative":
                    volunteer_part = """
                    SELECT 
                        vi.id,
                        vi.name_ru AS name,
                        vi.description,
                        'volunteer_initiative' AS type,
                        vi.feature_data AS features,
                        eg.geographical_entity_id,
                        ge.name_ru AS geo_name  -- ДОБАВЛЕНО: название географического объекта
                    FROM volunteer_initiative vi
                    JOIN entity_geo eg ON vi.id = eg.entity_id 
                        AND eg.entity_type = 'volunteer_initiative'
                    JOIN geographical_entity ge ON eg.geographical_entity_id = ge.id  -- ДОБАВЛЕНО: джойн с географическими объектами
                    """
                    entities_parts.append(volunteer_part)
                
                # Комбинируем все части запроса
                entities_query = " UNION ALL ".join(entities_parts)
                
                # Условие для фильтрации по подтипу (без изменений)
                subtype_condition = ""
                if object_subtype:
                    if object_type == "biological_entity" or object_type is None:
                        subtype_condition = "AND entities.type = %(object_subtype)s"
                        params['object_subtype'] = object_subtype
                    else:
                        subtype_condition = """
                        AND (
                            entities.features->'geo_type'->'specific_types' ? %(object_subtype)s
                            OR 
                            entities.features->'geo_type'->'primary_type' ? %(object_subtype)s
                            OR
                            entities.features->>'information_type' ILIKE %(object_subtype)s
                            OR
                            entities.features->>'type' ILIKE %(object_subtype)s
                            OR
                            entities.features->>'category' ILIKE %(object_subtype)s
                            OR
                            entities.features->>'flora_type' ILIKE %(object_subtype)s
                            OR
                            entities.features->>'fauna_type' ILIKE %(object_subtype)s
                        )
                        """
                        params['object_subtype'] = object_subtype
                
                final_query = query.format(
                    entities_query=entities_query,
                    subtype_condition=subtype_condition
                )
                
                logger.debug(f"Выполняется поиск объектов в полигоне с параметрами:")
                logger.debug(f"  - object_type: {object_type}")
                logger.debug(f"  - object_subtype: {object_subtype}")
                logger.debug(f"  - buffer_radius_km: {buffer_radius_km}")
                logger.debug(f"  - limit: {limit}")
                
                cursor.execute(final_query, params)
                results = cursor.fetchall()
                
                # Форматируем результаты
                formatted_results = []
                for row in results:
                    formatted_row = dict(row)
                    # Преобразование GeoJSON в Python-объект, если нужно
                    if isinstance(formatted_row['geojson'], str):
                        formatted_row['geojson'] = json.loads(formatted_row['geojson'])
                    
                    # Добавляем features в результат для дальнейшей обработки
                    if 'features' not in formatted_row:
                        formatted_row['features'] = {}
                    
                    # ДОБАВЛЕНО: Если geo_name отсутствует, используем name как fallback
                    if 'geo_name' not in formatted_row or not formatted_row['geo_name']:
                        formatted_row['geo_name'] = formatted_row.get('name', 'Неизвестная локация')
                    
                    formatted_results.append(formatted_row)
                
                logger.debug(f"Найдено объектов: {len(formatted_results)}")
                return formatted_results
                
        except Exception as e:
            logger.error(f"Ошибка поиска объектов по полигону: {str(e)}", exc_info=True)
            return []
        finally:
            conn.close()
                    
    def get_radius_intersection(
        self, 
        latitude: float, 
        longitude: float, 
        radius_km: float = 10.0
    ) -> Optional[Dict]:
        """Возвращает пересечение круга с заданным радиусом с полигонами и регионы"""
        conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        try:
            with conn.cursor() as cursor:
                query = """
                        WITH user_point AS (
                            SELECT ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography AS point_geog
                        ),
                        radius_circle AS (
                            SELECT ST_Buffer(point_geog, %s * 1000) AS circle_geog
                            FROM user_point
                        ),  
                        circle_geom AS (
                            SELECT ST_Transform(ST_SetSRID(circle_geog::geometry, 4326), 4326) AS circle_geom
                            FROM radius_circle
                        ),
                        -- Регионы, попавшие в радиус
                        regions_in_radius AS (
                            SELECT DISTINCT ge.id, ge.name_ru
                            FROM geographical_entity ge
                            JOIN entity_geo eg ON ge.id = eg.geographical_entity_id
                            JOIN map_content mc ON mc.id = eg.entity_id AND eg.entity_type = 'map_content'
                            CROSS JOIN circle_geom c
                            WHERE ST_Intersects(mc.geometry, c.circle_geom)
                        )
                        SELECT 
                            ST_AsGeoJSON(
                                ST_Intersection(
                                    ST_Union(mc.geometry), 
                                    (SELECT circle_geom FROM circle_geom)
                                )
                            )::json AS intersection_geojson,
                            COALESCE(
                                json_agg(
                                    json_build_object('id', r.id, 'name', r.name_ru)
                                ) FILTER (WHERE r.id IS NOT NULL),
                                '[]'::json
                            ) AS regions
                        FROM map_content mc
                        LEFT JOIN regions_in_radius r ON TRUE
                        WHERE ST_Intersects(
                            mc.geometry, 
                            (SELECT circle_geom FROM circle_geom)
                        )
                        GROUP BY r.id, r.name_ru;
                        """
                cursor.execute(query, (longitude, latitude, radius_km))
                result = cursor.fetchone()
                
                if result:
                    return {
                        "intersection": result['intersection_geojson'],
                        "regions": result['regions'] 
                    }
                return None
        except Exception as e:
            logger.error(f"Ошибка вычисления пересечения: {str(e)}")
            return None
        finally:
            conn.close()