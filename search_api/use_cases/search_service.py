# search_api/use_cases/search_service.py
from dataclasses import dataclass
from typing import Dict, Any
from ..domain.entities import SearchRequest, SearchResponse
from ..adapters.search_repository import SearchRepository


@dataclass
class SearchUseCase:
    _repository: SearchRepository

    def execute(self, request: SearchRequest) -> SearchResponse:
        objects = self._repository.search_objects(request.query)
        if not objects:
            return SearchResponse(
                query=request.query,
                modality_filter=request.modality_type,
                objects=[],
                resources=[]
            )
        
        object_ids = [obj.id for obj in objects]
        resources = self._repository.search_resources(object_ids, request.modality_type)
        
        return SearchResponse(
            query=request.query,
            modality_filter=request.modality_type,
            objects=objects,
            resources=resources
        )