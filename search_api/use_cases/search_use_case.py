import time
from dataclasses import dataclass
from typing import List, Optional

from ..domain.entities import SearchRequest, SearchResponse, ResourceCriteria, ObjectResult, ResourceResult
from ..adapters.search_repository import SearchRepository


@dataclass
class SearchUseCase:
    _repository: SearchRepository

    def execute(self, request: SearchRequest) -> SearchResponse:
        start_time = time.time()
        debug = {}

        objects: List[ObjectResult] = []
        object_ids: Optional[List[int]] = None

        if request.object:
            obj_start = time.time()
            objects = self._repository.find_objects_by_criteria(
                request.object, limit=request.limit, offset=request.offset
            )
            debug['objects_query_time'] = time.time() - obj_start
            object_ids = [obj.id for obj in objects] if objects else None

        resource_criteria = request.resource if request.resource else ResourceCriteria()
        res_start = time.time()
        resources = self._repository.find_resources_by_criteria(
            resource_criteria, object_ids, limit=request.limit, offset=request.offset
        )
        debug['resources_query_time'] = time.time() - res_start
        debug['total_time'] = time.time() - start_time

        response = SearchResponse(
            object_criteria=request.object,
            resource_criteria=request.resource,
            modality_filter=request.modality_type,
            objects=objects,
            resources=resources,
        )
        
        if request.debug:
            response.debug_info = debug
            
        return response