from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import hashlib


class ModalityType(Enum):
    TEXT = "Текст"
    IMAGE = "Изображение"
    GEODATA = "Геоданные"


@dataclass(frozen=True)
class DbId:
    value: str
    
    def __str__(self) -> str:
        return self.value
    
    def __hash__(self) -> int:
        return hash(self.value)


@dataclass
class ObjectType:
    id: Optional[int] = None
    name: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Object:
    db_id: DbId
    object_type_id: int
    object_properties: Dict[str, Any]
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ObjectNameSynonym:
    synonym: str
    language: str = 'ru'
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class BibliographicData:
    author_id: Optional[int] = None
    date: Optional[str] = None
    source_id: Optional[int] = None
    reliability_level_id: Optional[int] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class CreationData:
    creation_type: Optional[str] = None
    creation_tool: Optional[str] = None
    creation_params: Optional[Dict] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ResourceStatic:
    static_id: Optional[str]
    bibliographic_id: int
    creation_id: int
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class SupportMetadata:
    parameters: Dict[str, Any]
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_resource(cls, resource: Dict, resource_hash: Optional[str] = None) -> 'SupportMetadata':
        params = {'original_data': resource}
        if resource_hash:
            params['resource_hash'] = resource_hash
        return cls(parameters=params)


@dataclass
class Resource:
    title: Optional[str]
    uri: Optional[str]
    features: Optional[Dict[str, Any]]
    text_id: Optional[str]
    resource_static_id: int
    support_metadata_id: int
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Modality:
    modality_type: str
    value_table_name: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class TextValue:
    structured_data: Dict[str, Any]
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ImageValue:
    url: Optional[str] = None
    file_path: Optional[str] = None
    format: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class GeodataValue:
    geometry: Dict[str, Any]
    geometry_type: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Author:
    name: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class Source:
    name: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class ReliabilityLevel:
    name: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class ResourceImportResult:
    success_count: int = 0
    skipped_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            'success': self.success_count,
            'skipped': self.skipped_count,
            'errors': self.error_count
        }