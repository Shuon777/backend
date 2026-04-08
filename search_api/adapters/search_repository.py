# search_api/adapters/search_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from ..domain.entities import ObjectResult, ResourceResult


class SearchRepository(ABC):
    @abstractmethod
    def search_objects(self, query: str) -> List[ObjectResult]:
        pass

    @abstractmethod
    def search_resources(self, object_ids: List[int], modality: Optional[str]) -> List[ResourceResult]:
        pass