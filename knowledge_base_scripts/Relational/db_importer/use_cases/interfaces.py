"""Repository and service interfaces for use cases."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from ..domain.entities import (
    Resource,
    ObjectDescription,
    ObjectSynonym,
    ObjectProperty,
    PropertyValue,
    BibliographicData,
    GenerationData,
    SupportMetadata,
    TextModality,
    ImageModality,
    GeodataModality,
)


class ResourceRepository(ABC):
    """Repository interface for Resource entity."""
    
    @abstractmethod
    def resource_exists_by_hash(self, resource_hash: str) -> bool:
        """Check if resource with given hash exists."""
        pass
    
    @abstractmethod
    def save_resource(self, resource: Resource) -> int:
        """Save resource and return its ID."""
        pass
    
    @abstractmethod
    def link_resource_to_object(self, resource_id: int, object_description_id: int) -> None:
        """Link resource to object description."""
        pass


class ObjectDescriptionRepository(ABC):
    """Repository interface for ObjectDescription entity."""
    
    @abstractmethod
    def find_by_canonical_id(self, canonical_id: str, object_type: str) -> Optional[ObjectDescription]:
        """Find object description by canonical ID."""
        pass
    
    @abstractmethod
    def save(self, object_description: ObjectDescription) -> ObjectDescription:
        """Save object description."""
        pass
    
    @abstractmethod
    def add_synonym(self, synonym: ObjectSynonym) -> None:
        """Add synonym to object description."""
        pass
    
    @abstractmethod
    def add_property(self, property_obj: ObjectProperty) -> None:
        """Add property to object description."""
        pass


class PropertyValueRepository(ABC):
    """Repository interface for PropertyValue."""
    
    @abstractmethod
    def get_or_create(self, value: str) -> PropertyValue:
        """Get existing property value or create new one."""
        pass


class ModalityRepository(ABC):
    """Repository interface for modalities."""
    
    @abstractmethod
    def get_or_create_modality(self, modality_type: str) -> int:
        """Get or create modality and return its ID."""
        pass
    
    @abstractmethod
    def save_text_modality(self, modality: TextModality) -> None:
        """Save text modality data."""
        pass
    
    @abstractmethod
    def save_image_modality(self, modality: ImageModality) -> None:
        """Save image modality data."""
        pass
    
    @abstractmethod
    def save_geodata_modality(self, modality: GeodataModality) -> None:
        """Save geodata modality data."""
        pass


class BibliographicRepository(ABC):
    """Repository interface for BibliographicData."""
    
    @abstractmethod
    def get_or_create(self, bibliographic: BibliographicData) -> int:
        """Get existing bibliographic data or create new one."""
        pass


class GenerationRepository(ABC):
    """Repository interface for GenerationData."""
    
    @abstractmethod
    def get_or_create(self, generation: GenerationData) -> int:
        """Get existing generation data or create new one."""
        pass


class SupportMetadataRepository(ABC):
    """Repository interface for SupportMetadata."""
    
    @abstractmethod
    def get_or_create(self, metadata: SupportMetadata) -> int:
        """Get existing metadata or create new one."""
        pass
    
    @abstractmethod
    def update_hash(self, metadata_id: int, resource_hash: str) -> None:
        """Update resource hash in metadata."""
        pass


class SpeciesNameNormalizer(ABC):
    """Service interface for species name normalization."""
    
    @abstractmethod
    def normalize(self, name: str) -> str:
        """Normalize species name."""
        pass
    
class SchemaRepository(ABC):
    """Repository interface for schema management."""
    
    @abstractmethod
    def drop_all(self) -> None:
        """Drop all tables in schema."""
        pass
    
    @abstractmethod
    def create_all(self) -> None:
        """Create all tables from schema."""
        pass