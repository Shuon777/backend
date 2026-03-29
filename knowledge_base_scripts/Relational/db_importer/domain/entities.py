"""Domain entities for eco_assistant."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import hashlib


class ModalityType(Enum):
    """Types of modalities."""
    TEXT = "Текст"
    IMAGE = "Изображение"
    GEODATA = "Геоданные"
    AUDIO = "Аудио"


class ResourceType(Enum):
    """Types of resources."""
    IMAGE = "Изображение"
    TEXT = "Текст"
    MAP = "Картографическая информация"
    GEOGRAPHICAL_OBJECT = "Географический объект"


@dataclass(frozen=True)
class CanonicalId:
    """Canonical identifier for object (Value Object)."""
    value: str
    
    @classmethod
    def from_name_and_type(cls, name: str, object_type: str) -> 'CanonicalId':
        """Generate canonical ID from name and type."""
        normalized_name = name.strip().lower()
        combined = f"{normalized_name}|{object_type}"
        return cls(hashlib.md5(combined.encode('utf-8')).hexdigest())
    
    def __str__(self) -> str:
        return self.value


@dataclass
class ObjectDescription:
    """Object description entity."""
    canonical_id: CanonicalId
    object_type: str
    classification_identifier: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def add_synonym(self, synonym: str, language: str = 'ru', is_primary: bool = False) -> 'ObjectSynonym':
        """Create a synonym for this object."""
        return ObjectSynonym(
            object_description_id=self.id,
            synonym=synonym,
            language=language,
            is_primary=is_primary
        )


@dataclass
class ObjectSynonym:
    """Object synonym value object."""
    object_description_id: Optional[int]
    synonym: str
    language: str = 'ru'
    is_primary: bool = False
    id: Optional[int] = None


@dataclass
class PropertyValue:
    """Property value value object."""
    value: str
    id: Optional[int] = None
    
    def __hash__(self) -> int:
        return hash(self.value)


@dataclass
class ObjectProperty:
    """Object property value object."""
    object_description_id: int
    property_name: str
    object_type: str
    property_value: PropertyValue
    id: Optional[int] = None


@dataclass
class BibliographicData:
    """Bibliographic data value object."""
    author: Optional[str] = None
    date: Optional[str] = None
    source: Optional[str] = None
    rights: Optional[str] = None
    reliability: Optional[str] = None
    id: Optional[int] = None


@dataclass
class GenerationData:
    """Generation data value object."""
    generation_type: Optional[str] = None
    generation_tool: Optional[str] = None
    generation_params: Optional[Dict] = None
    id: Optional[int] = None


@dataclass
class SupportMetadata:
    """Support metadata value object."""
    parameters: Dict[str, Any]
    id: Optional[int] = None
    
    @classmethod
    def from_resource(cls, resource: Dict, resource_hash: Optional[str] = None) -> 'SupportMetadata':
        """Create metadata from resource data."""
        params = {'original_data': resource}
        if resource_hash:
            params['resource_hash'] = resource_hash
        return cls(parameters=params)


@dataclass
class Resource:
    """Resource entity."""
    modality_id: int
    bibliographic_id: int
    generation_id: int
    support_metadata_id: int
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class TextModality:
    """Text modality value object."""
    modality_id: int
    content: Dict[str, Any]
    id: Optional[int] = None


@dataclass
class ImageModality:
    """Image modality value object."""
    modality_id: int
    url: Optional[str] = None
    file_path: Optional[str] = None
    quality: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    id: Optional[int] = None


@dataclass
class GeodataModality:
    """Geodata modality value object."""
    modality_id: int
    geometry: Dict[str, Any]
    id: Optional[int] = None


@dataclass
class ResourceImportResult:
    """Result of batch import operation."""
    success_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            'success': self.success_count,
            'skipped': self.skipped_count,
            'errors': self.error_count
        }