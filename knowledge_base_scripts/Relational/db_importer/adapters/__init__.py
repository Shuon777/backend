"""Adapters layer - database clients and repository implementations."""

from .database_client import DatabaseClient, PostgresClient
from .postgres_repositories import (
    PostgresResourceRepository,
    PostgresObjectDescriptionRepository,
    PostgresPropertyValueRepository,
    PostgresModalityRepository,
    PostgresBibliographicRepository,
    PostgresGenerationRepository,
    PostgresSupportMetadataRepository,
)
from .schema_repository import PostgresSchemaRepository

__all__ = [
    'DatabaseClient',
    'PostgresClient',
    'PostgresResourceRepository',
    'PostgresObjectDescriptionRepository',
    'PostgresPropertyValueRepository',
    'PostgresModalityRepository',
    'PostgresBibliographicRepository',
    'PostgresGenerationRepository',
    'PostgresSupportMetadataRepository',
    'PostgresSchemaRepository',
]