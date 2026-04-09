# search_api/domain/entities.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ObjectCriteria:
    db_id: Optional[str] = None
    name_synonyms: Optional[Dict[str, List[str]]] = None
    properties: Optional[Dict[str, Any]] = None
    object_type: Optional[str] = None


@dataclass
class ResourceCriteria:
    title: Optional[str] = None
    uri: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    modality_type: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    structured_data: Optional[Dict[str, Any]] = None
    taxonomy: Optional[Dict[str, str]] = None


@dataclass
class SearchRequest:
    object: Optional[ObjectCriteria] = None
    resource: Optional[ResourceCriteria] = None
    modality_type: Optional[str] = None
    limit: int = 20
    offset: int = 0
    debug: bool = False
    search_type: str = "Relational"
    use_llm_answer: bool = False
    user_query: Optional[str] = None
    clean_user_query: Optional[str] = None


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
    author: Optional[str]
    source: Optional[str]
    modality_type: str
    content: Any
    features: Optional[Dict[str, Any]] = None


@dataclass
class SearchResponse:
    object_criteria: Optional[ObjectCriteria]
    resource_criteria: Optional[ResourceCriteria]
    modality_filter: Optional[str]
    objects: List[ObjectResult]
    resources: List[ResourceResult]
    debug_info: Optional[Dict[str, Any]] = None
    llm_answer: Optional[Dict[str, Any]] = None