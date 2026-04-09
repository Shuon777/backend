# search_api/use_cases/search_service.py
import time
from dataclasses import dataclass
from typing import List, Optional
from ..domain.entities import SearchRequest, SearchResponse, ResourceCriteria, ObjectResult, ResourceResult
from ..adapters.search_repository import SearchRepository
from ..services import LLMAnswerGenerator


@dataclass
class SearchUseCase:
    _repository: SearchRepository

    def execute(self, request: SearchRequest) -> SearchResponse:
        start_time = time.time()
        debug_info = {}
        llm_answer = None
        
        objects: List[ObjectResult] = []
        object_ids: Optional[List[int]] = None
        
        if request.object:
            obj_start = time.time()
            objects = self._repository.find_objects_by_criteria(
                request.object,
                limit=request.limit,
                offset=request.offset
            )
            debug_info['objects_query_time'] = time.time() - obj_start
            object_ids = [obj.id for obj in objects] if objects else None
        
        resource_criteria = request.resource if request.resource else ResourceCriteria()
        res_start = time.time()
        resources = self._repository.find_resources_by_criteria(
            resource_criteria,
            object_ids,
            limit=request.limit,
            offset=request.offset
        )
        debug_info['resources_query_time'] = time.time() - res_start
        debug_info['total_time'] = time.time() - start_time
        
        if request.use_llm_answer and request.user_query:
            query = request.clean_user_query if request.clean_user_query else request.user_query
            generator = LLMAnswerGenerator()
            llm_start = time.time()
            llm_answer = generator.generate(query, objects, resources)
            debug_info['llm_time'] = time.time() - llm_start
        
        response = SearchResponse(
            object_criteria=request.object,
            resource_criteria=request.resource,
            modality_filter=request.modality_type,
            objects=objects,
            resources=resources,
            llm_answer=llm_answer
        )
        if request.debug:
            response.debug_info = debug_info
        return response