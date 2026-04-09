# search_api/adapters/search_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from ..domain.entities import ObjectResult, ResourceResult, ObjectCriteria, ResourceCriteria


class SearchRepository(ABC):
    @abstractmethod
    def find_objects_by_criteria(self, criteria: ObjectCriteria, limit: int = 20, offset: int = 0) -> List[ObjectResult]:
        pass

    @abstractmethod
    def find_resources_by_criteria(self, criteria: ResourceCriteria, object_ids: Optional[List[int]] = None, limit: int = 50, offset: int = 0) -> List[ResourceResult]:
        pass