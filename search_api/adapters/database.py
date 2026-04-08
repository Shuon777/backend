# search_api/adapters/database.py
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict, Any
from ..config import SearchConfig
from ..domain.entities import ObjectResult, ResourceResult
from .search_repository import SearchRepository


class PostgresSearchRepository(SearchRepository):
    def __init__(self, config: SearchConfig):
        self._conn_params = {
            'dbname': config.db_name,
            'user': config.db_user,
            'password': config.db_password,
            'host': config.db_host,
            'port': config.db_port
        }

    def _get_conn(self):
        return psycopg2.connect(**self._conn_params, cursor_factory=RealDictCursor)

    def search_objects(self, query: str) -> List[ObjectResult]:
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
                    WHERE ons.synonym ILIKE %s
                    GROUP BY o.id, ot.name
                    LIMIT 20
                """
                cur.execute(sql, (f'%{query}%',))
                rows = cur.fetchall()
                return [
                    ObjectResult(
                        id=r['id'],
                        db_id=r['db_id'],
                        object_type=r['object_type'],
                        properties=r['object_properties'],
                        synonyms=r['synonyms'] or []
                    ) for r in rows
                ]

    def search_resources(self, object_ids: List[int], modality: Optional[str]) -> List[ResourceResult]:
        if not object_ids:
            return []
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT DISTINCT r.id, r.title, r.uri,
                           m.modality_type,
                           CASE 
                               WHEN m.modality_type = 'Текст' THEN 
                                   jsonb_build_object('structured_data', tv.structured_data)
                               WHEN m.modality_type = 'Изображение' THEN 
                                   jsonb_build_object('url', iv.url, 'file_path', iv.file_path, 'format', iv.format)
                               WHEN m.modality_type = 'Геоданные' THEN 
                                   jsonb_build_object('geojson', ST_AsGeoJSON(gv.geometry), 'type', gv.geometry_type)
                           END as content
                    FROM eco_assistant.resource r
                    JOIN eco_assistant.resource_object ro ON r.id = ro.resource_id
                    JOIN eco_assistant.resource_value rv ON r.id = rv.resource_id
                    JOIN eco_assistant.modality m ON rv.modality_id = m.id
                    LEFT JOIN eco_assistant.text_value tv ON rv.value_id = tv.id AND m.modality_type = 'Текст'
                    LEFT JOIN eco_assistant.image_value iv ON rv.value_id = iv.id AND m.modality_type = 'Изображение'
                    LEFT JOIN eco_assistant.geodata_value gv ON rv.value_id = gv.id AND m.modality_type = 'Геоданные'
                    WHERE ro.object_id = ANY(%s)
                """
                params = [object_ids]
                if modality:
                    sql += " AND m.modality_type = %s"
                    params.append(modality)
                sql += " LIMIT 50"
                cur.execute(sql, params)
                rows = cur.fetchall()
                return [
                    ResourceResult(
                        id=r['id'],
                        title=r['title'],
                        uri=r['uri'],
                        modality_type=r['modality_type'],
                        content=r['content']
                    ) for r in rows
                ]