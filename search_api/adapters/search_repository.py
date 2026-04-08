from abc import ABC, abstractmethod
from typing import List, Optional
from ..domain.entities import ObjectResult, ResourceResult, ObjectCriteria, ResourceCriteria


class SearchRepository(ABC):
    @abstractmethod
    def find_objects_by_criteria(self, criteria: ObjectCriteria) -> List[ObjectResult]:
        pass

    @abstractmethod
    def find_resources_by_criteria(self, criteria: ResourceCriteria, object_ids: Optional[List[int]] = None) -> List[ResourceResult]:
        pass