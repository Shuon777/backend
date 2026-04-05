import json
from typing import Optional, Dict, List, Any, Tuple
from psycopg2.extras import Json as PgJson

from ..domain.entities import (
    Object,
    ObjectNameSynonym,
    BibliographicData,
    CreationData,
    ResourceStatic,
    SupportMetadata,
    Resource,
    TextValue,
    ImageValue,
    GeodataValue,
    Modality,
    ObjectType,
    DbId,
    Author,
    Source,
    ReliabilityLevel,
)
from ..use_cases.interfaces import (
    ResourceRepository,
    ObjectRepository,
    ObjectTypeRepository,
    SynonymRepository,
    ModalityRepository,
    BibliographicRepository,
    CreationRepository,
    ResourceStaticRepository,
    SupportMetadataRepository,
)
from .database_client import DatabaseClient


class PostgresResourceRepository(ResourceRepository):
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
            "INSERT INTO eco_assistant.resource (title, uri, features, text_id, resource_static_id, support_metadata_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (resource.title, resource.uri, 
             PgJson(resource.features) if resource.features else None,
             resource.text_id,
             resource.resource_static_id, resource.support_metadata_id)
        )
        self._client.commit()
        return row[0]

    def link_resource_to_object(self, resource_id: int, object_id: int, relation_type: Optional[str] = None) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.resource_object (resource_id, object_id, relation_type) "
            "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (resource_id, object_id, relation_type)
        )
        self._client.commit()

    def find_resource_by_text_id(self, text_id: str) -> Optional[int]:
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.resource WHERE text_id = %s",
            (text_id,)
        )
        return row[0] if row else None

    def link_resource_to_resource(self, resource_id: int, related_resource_id: int, relation_type: str) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.resource_resource_link (resource_id, related_resource_id, relation_type) "
            "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (resource_id, related_resource_id, relation_type)
        )
        self._client.commit()


class PostgresObjectRepository(ObjectRepository):
    def __init__(self, client: DatabaseClient):
        self._client = client

    def find_by_db_id(self, db_id: str) -> Optional[Object]:
        row = self._client.fetchone(
            "SELECT id, db_id, object_type_id, object_properties, created_at, updated_at "
            "FROM eco_assistant.object WHERE db_id = %s",
            (db_id,)
        )
        if not row:
            return None
        
        return Object(
            id=row[0],
            db_id=DbId(row[1]),
            object_type_id=row[2],
            object_properties=row[3],
            created_at=row[4],
            updated_at=row[5]
        )
    
    def save(self, obj: Object) -> Object:
        if obj.id is not None:
            row = self._client.fetchone(
                "UPDATE eco_assistant.object SET object_type_id = %s, object_properties = %s, updated_at = now() "
                "WHERE id = %s RETURNING id, created_at, updated_at",
                (obj.object_type_id, PgJson(obj.object_properties), obj.id)
            )
            if row:
                obj.created_at = row[1]
                obj.updated_at = row[2]
        else:
            row = self._client.fetchone(
                "INSERT INTO eco_assistant.object (db_id, object_type_id, object_properties) "
                "VALUES (%s, %s, %s) RETURNING id, created_at, updated_at",
                (str(obj.db_id), obj.object_type_id, PgJson(obj.object_properties))
            )
            obj.id = row[0]
            obj.created_at = row[1]
            obj.updated_at = row[2]
        
        self._client.commit()
        return obj

    def find_by_db_id_only(self, db_id: str) -> Optional[Object]:
        row = self._client.fetchone(
            "SELECT id, db_id, object_type_id, object_properties, created_at, updated_at "
            "FROM eco_assistant.object WHERE db_id = %s",
            (db_id,)
        )
        if not row:
            return None
        return Object(
            id=row[0],
            db_id=DbId(row[1]),
            object_type_id=row[2],
            object_properties=row[3],
            created_at=row[4],
            updated_at=row[5]
        )

    def add_synonym_link(self, object_id: int, synonym_id: int) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.object_name_synonym_link (object_id, synonym_id) "
            "VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (object_id, synonym_id)
        )
        self._client.commit()

    def link_object_to_object(self, object_id: int, related_object_id: int, relation_type: str) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.object_object_link (object_id, related_object_id, relation_type) "
            "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (object_id, related_object_id, relation_type)
        )
        self._client.commit()


class PostgresObjectTypeRepository(ObjectTypeRepository):
    def __init__(self, client: DatabaseClient):
        self._client = client
        self._cache: Dict[str, ObjectType] = {}

    def get_or_create(self, name: str) -> ObjectType:
        if name in self._cache:
            return self._cache[name]

        row = self._client.fetchone(
            "SELECT id, name, schema FROM eco_assistant.object_type WHERE name = %s",
            (name,)
        )
        if row:
            obj_type = ObjectType(id=row[0], name=row[1], schema=row[2])
            self._cache[name] = obj_type
            return obj_type

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.object_type (name, schema) VALUES (%s, %s) RETURNING id",
            (name, PgJson({}))
        )
        obj_type = ObjectType(id=row[0], name=name, schema={})
        self._client.commit()
        self._cache[name] = obj_type
        return obj_type


class PostgresSynonymRepository(SynonymRepository):
    def __init__(self, client: DatabaseClient):
        self._client = client
        self._cache: Dict[str, ObjectNameSynonym] = {}

    def get_or_create(self, synonym: str, language: str) -> ObjectNameSynonym:
        key = f"{synonym}|{language}"
        if key in self._cache:
            return self._cache[key]

        row = self._client.fetchone(
            "SELECT id, synonym, language FROM eco_assistant.object_name_synonym "
            "WHERE synonym = %s AND language = %s",
            (synonym, language)
        )
        if row:
            syn = ObjectNameSynonym(
                id=row[0],
                synonym=row[1],
                language=row[2]
            )
            self._cache[key] = syn
            return syn

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.object_name_synonym (synonym, language) "
            "VALUES (%s, %s) RETURNING id",
            (synonym, language)
        )
        syn = ObjectNameSynonym(
            id=row[0],
            synonym=synonym,
            language=language
        )
        self._client.commit()
        self._cache[key] = syn
        return syn


class PostgresModalityRepository(ModalityRepository):
    def __init__(self, client: DatabaseClient):
        self._client = client
        self._cache: Dict[str, Modality] = {}

    def get_or_create_modality(self, modality_type: str, value_table_name: str) -> Modality:
        if modality_type in self._cache:
            return self._cache[modality_type]

        row = self._client.fetchone(
            "SELECT id, modality_type, value_table_name FROM eco_assistant.modality "
            "WHERE modality_type = %s",
            (modality_type,)
        )
        if row:
            mod = Modality(
                id=row[0],
                modality_type=row[1],
                value_table_name=row[2]
            )
            self._cache[modality_type] = mod
            return mod

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.modality (modality_type, value_table_name) "
            "VALUES (%s, %s) RETURNING id",
            (modality_type, value_table_name)
        )
        mod = Modality(
            id=row[0],
            modality_type=modality_type,
            value_table_name=value_table_name
        )
        self._client.commit()
        self._cache[modality_type] = mod
        return mod

    def save_text_value(self, value: TextValue) -> int:
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.text_value (structured_data) VALUES (%s) RETURNING id",
            (PgJson(value.structured_data),)
        )
        self._client.commit()
        return row[0]

    def save_image_value(self, value: ImageValue) -> int:
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.image_value (url, file_path, format) "
            "VALUES (%s, %s, %s) RETURNING id",
            (value.url, value.file_path, value.format)
        )
        self._client.commit()
        return row[0]

    def save_geodata_value(self, value: GeodataValue) -> int:
        geom_json = json.dumps(value.geometry)
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.geodata_value (geometry, geometry_type) "
            "VALUES (ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s) RETURNING id",
            (geom_json, value.geometry_type)
        )
        self._client.commit()
        return row[0]

    def link_resource_value(self, resource_id: int, modality_id: int, value_id: Optional[int]) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.resource_value (resource_id, modality_id, value_id) "
            "VALUES (%s, %s, %s) ON CONFLICT (resource_id, modality_id) DO NOTHING",
            (resource_id, modality_id, value_id)
        )
        self._client.commit()


class PostgresBibliographicRepository(BibliographicRepository):
    def __init__(self, client: DatabaseClient):
        self._client = client
        self._author_cache: Dict[str, int] = {}
        self._source_cache: Dict[str, int] = {}
        self._reliability_cache: Dict[str, int] = {}

    def get_or_create_author(self, name: str) -> int:
        if not name:
            return None
        if name in self._author_cache:
            return self._author_cache[name]

        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.author WHERE name = %s",
            (name,)
        )
        if row:
            self._author_cache[name] = row[0]
            return row[0]

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.author (name) VALUES (%s) RETURNING id",
            (name,)
        )
        self._client.commit()
        self._author_cache[name] = row[0]
        return row[0]

    def get_or_create_source(self, name: str) -> int:
        if not name:
            return None
        if name in self._source_cache:
            return self._source_cache[name]

        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.source WHERE name = %s",
            (name,)
        )
        if row:
            self._source_cache[name] = row[0]
            return row[0]

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.source (name) VALUES (%s) RETURNING id",
            (name,)
        )
        self._client.commit()
        self._source_cache[name] = row[0]
        return row[0]

    def get_or_create_reliability_level(self, name: str) -> int:
        if not name:
            return None
        if name in self._reliability_cache:
            return self._reliability_cache[name]

        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.reliability_level WHERE name = %s",
            (name,)
        )
        if row:
            self._reliability_cache[name] = row[0]
            return row[0]

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.reliability_level (name) VALUES (%s) RETURNING id",
            (name,)
        )
        self._client.commit()
        self._reliability_cache[name] = row[0]
        return row[0]

    def get_or_create(self, bibliographic: BibliographicData) -> int:
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.bibliographic WHERE "
            "COALESCE(author_id, 0) = COALESCE(%s, 0) AND "
            "COALESCE(date::text, '') = COALESCE(%s, '') AND "
            "COALESCE(source_id, 0) = COALESCE(%s, 0) AND "
            "COALESCE(reliability_level_id, 0) = COALESCE(%s, 0)",
            (bibliographic.author_id, bibliographic.date, bibliographic.source_id,
             bibliographic.reliability_level_id)
        )
        if row:
            return row[0]

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.bibliographic "
            "(author_id, date, source_id, reliability_level_id) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (bibliographic.author_id, bibliographic.date if bibliographic.date else None,
             bibliographic.source_id, bibliographic.reliability_level_id)
        )
        self._client.commit()
        return row[0]


class PostgresCreationRepository(CreationRepository):
    def __init__(self, client: DatabaseClient):
        self._client = client
        self._cache: Dict[tuple, int] = {}

    def get_or_create(self, creation: CreationData) -> int:
        key = (creation.creation_type or '', creation.creation_tool or '',
               json.dumps(creation.creation_params or {}, sort_keys=True))

        if key in self._cache:
            return self._cache[key]

        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.creation WHERE "
            "COALESCE(creation_type,'') = %s AND COALESCE(creation_tool,'') = %s "
            "AND COALESCE(creation_params::text,'') = %s",
            (creation.creation_type, creation.creation_tool,
             json.dumps(creation.creation_params or {}))
        )
        if row:
            self._cache[key] = row[0]
            return row[0]

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.creation (creation_type, creation_tool, creation_params) "
            "VALUES (%s, %s, %s) RETURNING id",
            (creation.creation_type, creation.creation_tool,
             PgJson(creation.creation_params) if creation.creation_params else None)
        )
        self._client.commit()
        self._cache[key] = row[0]
        return row[0]


class PostgresResourceStaticRepository(ResourceStaticRepository):
    def __init__(self, client: DatabaseClient):
        self._client = client
        self._cache: Dict[tuple, int] = {}

    def find_by_static_id(self, static_id: str) -> Optional[int]:
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.resource_static WHERE static_id = %s",
            (static_id,)
        )
        return row[0] if row else None

    def get_or_create(self, static: ResourceStatic) -> int:
        key = (static.static_id, static.bibliographic_id, static.creation_id)
        if key in self._cache:
            return self._cache[key]

        # Try to find by static_id first
        if static.static_id:
            row = self._client.fetchone(
                "SELECT id FROM eco_assistant.resource_static WHERE static_id = %s",
                (static.static_id,)
            )
            if row:
                self._cache[key] = row[0]
                return row[0]

        # Find by bibliographic and creation
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.resource_static WHERE "
            "bibliographic_id = %s AND creation_id = %s",
            (static.bibliographic_id, static.creation_id)
        )
        if row:
            self._cache[key] = row[0]
            return row[0]

        # Create new
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.resource_static (static_id, bibliographic_id, creation_id) "
            "VALUES (%s, %s, %s) RETURNING id",
            (static.static_id, static.bibliographic_id, static.creation_id)
        )
        self._client.commit()
        self._cache[key] = row[0]
        return row[0]


class PostgresSupportMetadataRepository(SupportMetadataRepository):
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
            self._cache[key] = row[0]
            return row[0]

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.support_metadata (parameters) VALUES (%s) RETURNING id",
            (PgJson(metadata.parameters),)
        )
        self._client.commit()
        self._cache[key] = row[0]
        return row[0]

    def update_hash(self, metadata_id: int, resource_hash: str) -> None:
        self._client.execute(
            "UPDATE eco_assistant.support_metadata SET parameters = parameters || %s WHERE id = %s",
            (PgJson({'resource_hash': resource_hash}), metadata_id)
        )
        self._client.commit()