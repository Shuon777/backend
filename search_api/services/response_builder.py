from typing import Dict, Any, List, Optional

from ..domain.entities import ObjectResult, ResourceResult, SearchResponse
from ..domain.value_objects import ModalityType, GeoContent
from .geo_map_service import GeoMapService
from .llm_answer_generator import LLMAnswerGenerator


class ResponseBuilder:
    def __init__(self, geo_service: GeoMapService, llm_generator: LLMAnswerGenerator):
        self._geo_service = geo_service
        self._llm_generator = llm_generator

    def build(self, search_response: SearchResponse, user_query: Optional[str] = None,
              use_llm: bool = False) -> Dict[str, Any]:
        result = {
            'object_criteria': self._serialize_object_criteria(search_response.object_criteria),
            'resource_criteria': self._serialize_resource_criteria(search_response.resource_criteria),
            'modality_filter': search_response.modality_filter,
            'objects': self._serialize_objects(search_response.objects),
            'resources': self._serialize_resources(search_response.resources),
        }
        if search_response.debug_info:
            result['debug'] = search_response.debug_info
        if use_llm and user_query and search_response.resources:
            llm_answer = self._llm_generator.generate(user_query, search_response.objects, search_response.resources)
            result['llm_answer'] = llm_answer
        return result

    def _serialize_object_criteria(self, criteria) -> Optional[Dict[str, Any]]:
        if not criteria:
            return None
        return {
            'db_id': criteria.db_id,
            'name_synonyms': criteria.name_synonyms,
            'properties': criteria.properties,
            'object_type': criteria.object_type,
        }

    def _serialize_resource_criteria(self, criteria) -> Optional[Dict[str, Any]]:
        if not criteria:
            return None
        return {
            'title': criteria.title,
            'author': criteria.author,
            'source': criteria.source,
            'modality_type': criteria.modality_type,
            'features': criteria.features,
        }

    def _serialize_objects(self, objects: List[ObjectResult]) -> List[Dict[str, Any]]:
        return [
            {
                'id': o.id,
                'db_id': o.db_id,
                'type': o.object_type,
                'properties': o.properties,
                'synonyms': o.synonyms,
            }
            for o in objects
        ]

    def _serialize_resources(self, resources: List[ResourceResult]) -> List[Dict[str, Any]]:
        serialized = []
        for r in resources:
            item = {
                'id': r.id,
                'title': r.title,
                'uri': r.uri,
                'author': r.author,
                'source': r.source,
                'modality_type': r.modality_type,
                'features': r.features,
            }
            if r.modality_type == ModalityType.GEODATA.value:
                if isinstance(r.content, GeoContent):
                    item['content'] = {
                        'geojson': r.content.geojson,
                        'geometry_type': r.content.geometry_type,
                        'map_links': {
                            'static': r.content.map_links.static,
                            'interactive': r.content.map_links.interactive,
                        }
                    }
                else:
                    enriched = self._geo_service.enrich_geo_content(r.content, f"map_{r.id}")
                    item['content'] = {
                        'geojson': enriched.geojson,
                        'geometry_type': enriched.geometry_type,
                        'map_links': {
                            'static': enriched.map_links.static,
                            'interactive': enriched.map_links.interactive,
                        }
                    }
            else:
                item['content'] = r.content
            serialized.append(item)
        return serialized