from dataclasses import dataclass
from ..domain.entities import SearchRequest, SearchResponse, ResourceCriteria
from ..adapters.search_repository import SearchRepository


@dataclass
class SearchUseCase:
    _repository: SearchRepository

    def execute(self, request: SearchRequest) -> SearchResponse:
        objects = []
        object_ids = None
        
        if request.object:
            objects = self._repository.find_objects_by_criteria(request.object)
            object_ids = [obj.id for obj in objects] if objects else None
        
        resource_criteria = request.resource if request.resource else ResourceCriteria()
        resources = self._repository.find_resources_by_criteria(resource_criteria, object_ids)
        
        return SearchResponse(
            object_criteria=request.object,
            resource_criteria=request.resource,
            modality_filter=request.modality_type,
            objects=objects,
            resources=resources
        )