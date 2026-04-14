# search_api/adapters/database.py
import logging
import psycopg2
import json
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict, Any
from ..config import SearchConfig
from ..domain.entities import ObjectResult, ResourceResult, ObjectCriteria, ResourceCriteria
from .search_repository import SearchRepository

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True

class PostgresSearchRepository(SearchRepository):
    def __init__(self, config: SearchConfig):
        logger.debug(f"Initializing PostgresSearchRepository with config: {config.db_host}:{config.db_port}/{config.db_name}")
        self._conn_params = {
            'dbname': config.db_name,
            'user': config.db_user,
            'password': config.db_password,
            'host': config.db_host,
            'port': config.db_port
        }

    def _get_conn(self):
        logger.debug("Creating database connection")
        return psycopg2.connect(**self._conn_params, cursor_factory=RealDictCursor)

    def find_objects_by_criteria(self, criteria: ObjectCriteria, limit: int = 20, offset: int = 0) -> List[ObjectResult]:
        logger = logging.getLogger(__name__)
        
        logger.info(f"find_objects_by_criteria START")
        logger.info(f"criteria.properties: {criteria.properties}")
        
        if not criteria.db_id and not criteria.name_synonyms and not criteria.properties and not criteria.object_type:
            logger.info("No criteria provided, returning empty list")
            return []
        
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT DISTINCT o.id, o.db_id, ot.name as object_type,
                           o.object_properties,
                           array_agg(DISTINCT ons.synonym) FILTER (WHERE ons.synonym IS NOT NULL) as synonyms
                    FROM eco_assistant.object o
                    JOIN eco_assistant.object_type ot ON o.object_type_id = ot.id
                    LEFT JOIN eco_assistant.object_name_synonym_link osl ON o.id = osl.object_id
                    LEFT JOIN eco_assistant.object_name_synonym ons ON osl.synonym_id = ons.id
                    WHERE 1=1
                """
                params = []
                
                if criteria.db_id:
                    sql += " AND o.db_id = %s"
                    params.append(criteria.db_id)
                
                if criteria.object_type:
                    sql += " AND ot.name = %s"
                    params.append(criteria.object_type)
                
                if criteria.name_synonyms:
                    names = []
                    for lang, name_list in criteria.name_synonyms.items():
                        if name_list:
                            names.extend(name_list)
                    if names:
                        placeholders = ','.join(['%s'] * len(names))
                        sql += f" AND ons.synonym IN ({placeholders})"
                        params.extend(names)
                
                if criteria.properties:
                    for key, value in criteria.properties.items():
                        logger.info(f"Processing {key} = {value}")
                        
                        if isinstance(value, str):
                            sql += f" AND o.object_properties->'{key}' @> '\"{value}\"'::jsonb"
                        elif isinstance(value, list):
                            for item in value:
                                sql += f" AND o.object_properties->'{key}' @> '\"{item}\"'::jsonb"
                        elif isinstance(value, bool):
                            sql += f" AND (o.object_properties->>'{key}')::boolean = {str(value).lower()}"
                        elif isinstance(value, (int, float)):
                            sql += f" AND (o.object_properties->>'{key}')::numeric = {value}"
                        else:
                            sql += f" AND o.object_properties->>'{key}' = '{str(value)}'"
                
                sql += " GROUP BY o.id, ot.name"
                sql += f" LIMIT {limit} OFFSET {offset}"
                
                logger.info(f"SQL: {sql}")
                logger.info(f"Params: {params}")
                
                cur.execute(sql, params)
                rows = cur.fetchall()
                
                logger.info(f"Found {len(rows)} rows")
                
                return [
                    ObjectResult(
                        id=r['id'],
                        db_id=r['db_id'],
                        object_type=r['object_type'],
                        properties=r['object_properties'],
                        synonyms=r['synonyms'] or []
                    ) for r in rows
                ]
                                                                          
    def find_resources_by_criteria(self, criteria: ResourceCriteria, object_ids: Optional[List[int]] = None, limit: int = 50, offset: int = 0) -> List[ResourceResult]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT DISTINCT r.id, r.title, r.uri,
                           a.name as author,
                           s.name as source,
                           m.modality_type,
                           r.features,
                           CASE 
                               WHEN m.modality_type = 'Текст' THEN 
                                   jsonb_build_object('structured_data', tv.structured_data)
                               WHEN m.modality_type = 'Изображение' THEN 
                                   jsonb_build_object('url', iv.url, 'file_path', iv.file_path, 'format', iv.format)
                               WHEN m.modality_type = 'Геоданные' THEN 
                                   jsonb_build_object('geojson', ST_AsGeoJSON(gv.geometry), 'type', gv.geometry_type)
                           END as content
                    FROM eco_assistant.resource r
                    JOIN eco_assistant.resource_static rs ON r.resource_static_id = rs.id
                    JOIN eco_assistant.bibliographic b ON rs.bibliographic_id = b.id
                    LEFT JOIN eco_assistant.author a ON b.author_id = a.id
                    LEFT JOIN eco_assistant.source s ON b.source_id = s.id
                    JOIN eco_assistant.resource_value rv ON r.id = rv.resource_id
                    JOIN eco_assistant.modality m ON rv.modality_id = m.id
                    LEFT JOIN eco_assistant.text_value tv ON rv.value_id = tv.id AND m.modality_type = 'Текст'
                    LEFT JOIN eco_assistant.image_value iv ON rv.value_id = iv.id AND m.modality_type = 'Изображение'
                    LEFT JOIN eco_assistant.geodata_value gv ON rv.value_id = gv.id AND m.modality_type = 'Геоданные'
                    WHERE 1=1
                """
                params = []
                
                if object_ids:
                    sql += " AND EXISTS (SELECT 1 FROM eco_assistant.resource_object ro WHERE ro.resource_id = r.id AND ro.object_id = ANY(%s))"
                    params.append(object_ids)
                
                if criteria.title:
                    sql += " AND r.title ILIKE %s"
                    params.append(f"%{criteria.title}%")
                
                if criteria.uri:
                    sql += " AND r.uri = %s"
                    params.append(criteria.uri)
                
                if criteria.author:
                    sql += " AND a.name ILIKE %s"
                    params.append(f"%{criteria.author}%")
                
                if criteria.source:
                    sql += " AND s.name ILIKE %s"
                    params.append(f"%{criteria.source}%")
                
                if criteria.modality_type:
                    sql += " AND m.modality_type = %s"
                    params.append(criteria.modality_type)
                
                if criteria.features:
                    for key, value in criteria.features.items():
                        sql += " AND r.features @> %s::jsonb"
                        params.append(json.dumps({key: value}))
                
                if criteria.structured_data:
                    for key, value in criteria.structured_data.items():
                        sql += " AND tv.structured_data @> %s::jsonb"
                        params.append(json.dumps({key: value}))
                
                if criteria.taxonomy:
                    for key, value in criteria.taxonomy.items():
                        sql += " AND tv.structured_data->'taxonomy'->>%s = %s"
                        params.extend([key, value])
                
                sql += f" LIMIT {limit} OFFSET {offset}"
                cur.execute(sql, params)
                rows = cur.fetchall()
                return [
                    ResourceResult(
                        id=r['id'],
                        title=r['title'],
                        uri=r['uri'],
                        author=r['author'],
                        source=r['source'],
                        modality_type=r['modality_type'],
                        content=r['content'],
                        features=r['features']
                    ) for r in rows
                ]