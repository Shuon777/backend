"""PostgreSQL implementations of repository interfaces."""

import json
from typing import Optional, Dict, List, Any
from psycopg2.extras import Json as PgJson

from ..domain.entities import (
    ObjectDescription,
    ObjectSynonym,
    PropertyValue,
    ObjectProperty,
    BibliographicData,
    GenerationData,
    SupportMetadata,
    Resource,
    TextModality,
    ImageModality,
    GeodataModality,
    CanonicalId,
)
from ..use_cases.interfaces import (
    ResourceRepository,
    ObjectDescriptionRepository,
    PropertyValueRepository,
    ModalityRepository,
    BibliographicRepository,
    GenerationRepository,
    SupportMetadataRepository,
)
from .database_client import DatabaseClient


class PostgresResourceRepository(ResourceRepository):
    """PostgreSQL implementation of ResourceRepository."""
    
    def __init__(self, client: DatabaseClient):
        self._client = client
    
    def resource_exists_by_hash(self, resource_hash: str) -> bool:
        row = self._client.fetchone(
            "SELECT 1 FROM eco_assistant.support_metadata WHERE parameters->>'resource_hash' = %s",
            (resource_hash,)
        )
        return row is not None
    
    def save_resource(self, resource: Resource) -> int:
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.resource (modality_id, bibliographic_id, generation_id, support_metadata_id) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (resource.modality_id, resource.bibliographic_id, 
             resource.generation_id, resource.support_metadata_id)
        )
        self._client.commit()
        return row[0]
    
    def link_resource_to_object(self, resource_id: int, object_description_id: int) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.resource_object (resource_id, object_description_id) "
            "VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (resource_id, object_description_id)
        )
        self._client.commit()


class PostgresObjectDescriptionRepository(ObjectDescriptionRepository):
    """PostgreSQL implementation of ObjectDescriptionRepository."""
    
    def __init__(self, client: DatabaseClient):
        self._client = client
    
    def find_by_canonical_id(self, canonical_id: str, object_type: str) -> Optional[ObjectDescription]:
        row = self._client.fetchone(
            "SELECT id, canonical_id, object_type, classification_identifier, created_at, updated_at "
            "FROM eco_assistant.object_description WHERE canonical_id = %s AND object_type = %s",
            (canonical_id, object_type)
        )
        if not row:
            return None
        
        return ObjectDescription(
            id=row[0],
            canonical_id=CanonicalId(row[1]),
            object_type=row[2],
            classification_identifier=row[3],
            created_at=row[4],
            updated_at=row[5]
        )
    
    def save(self, object_description: ObjectDescription) -> ObjectDescription:
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.object_description (canonical_id, object_type, classification_identifier) "
            "VALUES (%s, %s, %s) RETURNING id, created_at, updated_at",
            (str(object_description.canonical_id), object_description.object_type, 
             object_description.classification_identifier)
        )
        object_description.id = row[0]
        object_description.created_at = row[1]
        object_description.updated_at = row[2]
        self._client.commit()
        return object_description
    
    def add_synonym(self, synonym: ObjectSynonym) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.object_synonym (object_description_id, synonym, language, is_primary) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (object_description_id, synonym, language) DO NOTHING",
            (synonym.object_description_id, synonym.synonym, synonym.language, synonym.is_primary)
        )
        self._client.commit()
    
    def add_property(self, property_obj: ObjectProperty) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.object_property (object_description_id, property_name, object_type, property_value_id) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (property_obj.object_description_id, property_obj.property_name, 
             property_obj.object_type, property_obj.property_value.id)
        )
        self._client.commit()


class PostgresPropertyValueRepository(PropertyValueRepository):
    """PostgreSQL implementation of PropertyValueRepository."""
    
    def __init__(self, client: DatabaseClient):
        self._client = client
        self._cache: Dict[str, PropertyValue] = {}
    
    def get_or_create(self, value: str) -> PropertyValue:
        if value in self._cache:
            return self._cache[value]
        
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.property_value (value) VALUES (%s) "
            "ON CONFLICT (value_md5) DO NOTHING RETURNING id",
            (value,)
        )
        
        if row:
            pv = PropertyValue(value=value, id=row[0])
            self._cache[value] = pv
            self._client.commit()
            return pv
        
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.property_value WHERE value_md5 = md5(%s)",
            (value,)
        )
        pv = PropertyValue(value=value, id=row[0])
        self._cache[value] = pv
        return pv


class PostgresModalityRepository(ModalityRepository):
    """PostgreSQL implementation of ModalityRepository."""
    
    def __init__(self, client: DatabaseClient):
        self._client = client
        self._cache: Dict[str, int] = {}
    
    def get_or_create_modality(self, modality_type: str) -> int:
        if modality_type in self._cache:
            return self._cache[modality_type]
        
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.modality WHERE modality_type = %s",
            (modality_type,)
        )
        
        if row:
            mod_id = row[0]
            self._cache[modality_type] = mod_id
            return mod_id
        
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.modality (modality_type) VALUES (%s) RETURNING id",
            (modality_type,)
        )
        mod_id = row[0]
        self._client.commit()
        self._cache[modality_type] = mod_id
        return mod_id
    
    def save_text_modality(self, modality: TextModality) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.modality_text (modality_id, content) VALUES (%s, %s)",
            (modality.modality_id, PgJson(modality.content))
        )
        self._client.commit()
    
    def save_image_modality(self, modality: ImageModality) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.modality_image (modality_id, url, file_path, quality, width, height, format) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (modality.modality_id, modality.url, modality.file_path, 
             modality.quality, modality.width, modality.height, modality.format)
        )
        self._client.commit()
    
    def save_geodata_modality(self, modality: GeodataModality) -> None:
        geom_json = json.dumps(modality.geometry)
        self._client.execute(
            "INSERT INTO eco_assistant.modality_geodata (modality_id, geometry) "
            "VALUES (%s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))",
            (modality.modality_id, geom_json)
        )
        self._client.commit()


class PostgresBibliographicRepository(BibliographicRepository):
    """PostgreSQL implementation of BibliographicRepository."""
    
    def __init__(self, client: DatabaseClient):
        self._client = client
        self._cache: Dict[tuple, int] = {}
    
    def get_or_create(self, bibliographic: BibliographicData) -> int:
        key = (bibliographic.author or '', bibliographic.date or '', 
               bibliographic.source or '', bibliographic.rights or '', 
               bibliographic.reliability or '')
        
        if key in self._cache:
            return self._cache[key]
        
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.bibliographic WHERE "
            "COALESCE(author, '') = COALESCE(%s, '') AND "
            "COALESCE(date::text, '') = COALESCE(%s, '') AND "
            "COALESCE(source, '') = COALESCE(%s, '') AND "
            "COALESCE(rights, '') = COALESCE(%s, '') AND "
            "COALESCE(reliability, '') = COALESCE(%s, '')",
            (bibliographic.author, bibliographic.date, bibliographic.source,
             bibliographic.rights, bibliographic.reliability)
        )
        
        if row:
            bib_id = row[0]
            self._cache[key] = bib_id
            return bib_id
        
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.bibliographic (author, date, source, rights, reliability) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (bibliographic.author, bibliographic.date if bibliographic.date else None,
             bibliographic.source, bibliographic.rights, bibliographic.reliability)
        )
        bib_id = row[0]
        self._client.commit()
        self._cache[key] = bib_id
        return bib_id


class PostgresGenerationRepository(GenerationRepository):
    """PostgreSQL implementation of GenerationRepository."""
    
    def __init__(self, client: DatabaseClient):
        self._client = client
        self._cache: Dict[tuple, int] = {}
    
    def get_or_create(self, generation: GenerationData) -> int:
        key = (generation.generation_type or '', generation.generation_tool or '',
               json.dumps(generation.generation_params or {}, sort_keys=True))
        
        if key in self._cache:
            return self._cache[key]
        
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.generation WHERE "
            "COALESCE(generation_type,'') = %s AND COALESCE(generation_tool,'') = %s "
            "AND COALESCE(generation_params::text,'') = %s",
            (generation.generation_type, generation.generation_tool,
             json.dumps(generation.generation_params or {}))
        )
        
        if row:
            gen_id = row[0]
            self._cache[key] = gen_id
            return gen_id
        
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.generation (generation_type, generation_tool, generation_params) "
            "VALUES (%s, %s, %s) RETURNING id",
            (generation.generation_type, generation.generation_tool,
             PgJson(generation.generation_params) if generation.generation_params else None)
        )
        gen_id = row[0]
        self._client.commit()
        self._cache[key] = gen_id
        return gen_id


class PostgresSupportMetadataRepository(SupportMetadataRepository):
    """PostgreSQL implementation of SupportMetadataRepository."""
    
    def __init__(self, client: DatabaseClient):
        self._client = client
        self._cache: Dict[str, int] = {}
    
    def get_or_create(self, metadata: SupportMetadata) -> int:
        key = json.dumps(metadata.parameters, sort_keys=True)
        
        if key in self._cache:
            return self._cache[key]
        
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.support_metadata WHERE parameters::text = %s",
            (json.dumps(metadata.parameters),)
        )
        
        if row:
            meta_id = row[0]
            self._cache[key] = meta_id
            return meta_id
        
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.support_metadata (parameters) VALUES (%s) RETURNING id",
            (PgJson(metadata.parameters),)
        )
        meta_id = row[0]
        self._client.commit()
        self._cache[key] = meta_id
        return meta_id
    
    def update_hash(self, metadata_id: int, resource_hash: str) -> None:
        self._client.execute(
            "UPDATE eco_assistant.support_metadata SET parameters = parameters || %s WHERE id = %s",
            (PgJson({'resource_hash': resource_hash}), metadata_id)
        )
        self._client.commit()