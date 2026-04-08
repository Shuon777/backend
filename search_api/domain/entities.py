# search_api/domain/entities.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class SearchRequest:
    query: str
    modality_type: Optional[str] = None


@dataclass
class ObjectResult:
    id: int
    db_id: str
    object_type: str
    properties: Dict[str, Any]
    synonyms: List[str]


@dataclass
class ResourceResult:
    id: int
    title: Optional[str]
    uri: Optional[str]
    modality_type: str
    content: Any


@dataclass
class SearchResponse:
    query: str
    modality_filter: Optional[str]
    objects: List[ObjectResult]
    resources: List[ResourceResult]