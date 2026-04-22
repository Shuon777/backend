from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple

from ..domain.entities import (
    Resource,
    Object,
    ObjectNameSynonym,
    BibliographicData,
    CreationData,
    ResourceStatic,
    SupportMetadata,
    TextValue,
    ImageValue,
    GeodataValue,
    Modality,
    Author,
    Source,
    ReliabilityLevel,
    ObjectType,
)


class ResourceRepository(ABC):
    @abstractmethod
    def resource_exists_by_hash(self, resource_hash: str) -> bool:
        pass

    @abstractmethod
    def save_resource(self, resource: Resource) -> int:
        pass

    @abstractmethod
    def link_resource_to_object(self, resource_id: int, object_id: int, relation_type: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def find_resource_by_text_id(self, text_id: str) -> Optional[int]:
        pass

    @abstractmethod
    def link_resource_to_resource(self, resource_id: int, related_resource_id: int, relation_type: str) -> None:
        pass


class ObjectRepository(ABC):
    @abstractmethod
    def find_by_db_id(self, db_id: str) -> Optional[Object]:
        pass
    
    @abstractmethod
    def save(self, obj: Object) -> Object:
        pass
    
    @abstractmethod
    def add_synonym_link(self, object_id: int, synonym_id: int) -> None:
        pass
    
    @abstractmethod
    def link_object_to_object(self, object_id: int, related_object_id: int, relation_type: str) -> None:
        pass


class ObjectTypeRepository(ABC):
    @abstractmethod
    def get_or_create(self, name: str) -> ObjectType:
        pass


class SynonymRepository(ABC):
    @abstractmethod
    def get_or_create(self, synonym: str, language: str) -> ObjectNameSynonym:
        pass


class ModalityRepository(ABC):
    @abstractmethod
    def get_or_create_modality(self, modality_type: str, value_table_name: str) -> Modality:
        pass

    @abstractmethod
    def save_text_value(self, value: TextValue) -> int:
        pass

    @abstractmethod
    def save_image_value(self, value: ImageValue) -> int:
        pass

    @abstractmethod
    def save_geodata_value(self, value: GeodataValue) -> int:
        pass

    @abstractmethod
    def link_resource_value(self, resource_id: int, modality_id: int, value_id: Optional[int]) -> None:
        pass


class BibliographicRepository(ABC):
    @abstractmethod
    def get_or_create_author(self, name: str) -> int:
        pass

    @abstractmethod
    def get_or_create_source(self, name: str) -> int:
        pass

    @abstractmethod
    def get_or_create_reliability_level(self, name: str) -> int:
        pass

    @abstractmethod
    def get_or_create(self, bibliographic: BibliographicData) -> int:
        pass


class CreationRepository(ABC):
    @abstractmethod
    def get_or_create(self, creation: CreationData) -> int:
        pass


class ResourceStaticRepository(ABC):
    @abstractmethod
    def get_or_create(self, static: ResourceStatic) -> int:
        pass

    @abstractmethod
    def find_by_static_id(self, static_id: str) -> Optional[int]:
        pass


class SupportMetadataRepository(ABC):
    @abstractmethod
    def get_or_create(self, metadata: SupportMetadata) -> int:
        pass

    @abstractmethod
    def update_hash(self, metadata_id: int, resource_hash: str) -> None:
        pass


class SpeciesNameNormalizer(ABC):
    @abstractmethod
    def normalize(self, name: str) -> str:
        pass


class SchemaRepository(ABC):
    @abstractmethod
    def drop_all(self) -> None:
        pass

    @abstractmethod
    def create_all(self) -> None:
        pass


class GeodataProvider(ABC):
    @abstractmethod
    def get_geometry(self, geodb_id: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """Returns (geometry_dict, geometry_type) or None"""
        pass
    
class ObjectPropertyRepository(ABC):
    @abstractmethod
    def add_or_update_property(self, object_type_id: int, property_name: str, values: List[str]) -> None:
        pass

class ResourceFeatureRepository(ABC):
    @abstractmethod
    def add_or_update_feature(self, modality_id: int, feature_name: str, values: List[str]) -> None:
        pass