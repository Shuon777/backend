"""Main entry point for database importer."""

import sys
import argparse
import json
from pathlib import Path

# Исправляем на относительные импорты
from .config import DatabaseConfig
from .adapters import (
    PostgresClient,
    PostgresResourceRepository,
    PostgresObjectDescriptionRepository,
    PostgresPropertyValueRepository,
    PostgresModalityRepository,
    PostgresBibliographicRepository,
    PostgresGenerationRepository,
    PostgresSupportMetadataRepository,
    PostgresSchemaRepository,
)
from .services import JsonSpeciesNormalizer
from .use_cases import ImportResourceUseCase, BatchImportUseCase
from .infrastructure.logging_setup import setup_logging


def create_use_cases(config: DatabaseConfig, synonyms_path: Path):
    """Factory function to create use cases with dependencies."""
    
    client = PostgresClient(config)
    client.connect()
    
    # Create repositories
    resource_repo = PostgresResourceRepository(client)
    object_repo = PostgresObjectDescriptionRepository(client)
    property_value_repo = PostgresPropertyValueRepository(client)
    modality_repo = PostgresModalityRepository(client)
    bibliographic_repo = PostgresBibliographicRepository(client)
    generation_repo = PostgresGenerationRepository(client)
    metadata_repo = PostgresSupportMetadataRepository(client)
    
    # Create services
    species_normalizer = JsonSpeciesNormalizer(synonyms_path)
    
    # Create use cases
    import_resource = ImportResourceUseCase(
        resource_repo=resource_repo,
        object_repo=object_repo,
        property_value_repo=property_value_repo,
        modality_repo=modality_repo,
        bibliographic_repo=bibliographic_repo,
        generation_repo=generation_repo,
        metadata_repo=metadata_repo,
        species_normalizer=species_normalizer
    )
    
    batch_import = BatchImportUseCase(import_resource)
    
    return client, batch_import


def recreate_schema(client: PostgresClient, schema_file: Path) -> None:
    """Recreate database schema."""
    repo = PostgresSchemaRepository(client, schema_file)
    repo.drop_all()
    repo.create_all()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Database importer for eco_assistant schema')
    parser.add_argument('--full', action='store_true', help='Drop and recreate schema')
    parser.add_argument('--incremental', action='store_true', help='Import incrementally (skip duplicates)')
    parser.add_argument('--json-file', default='../../json_files/resources_dist.json', help='Path to JSON resources')
    parser.add_argument('--synonyms-file', default='json_files/object_synonyms.json', help='Path to synonyms file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--error-log', default='import_errors.log', help='File to log errors')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose, error_log=args.error_log)
    
    config = DatabaseConfig.from_env()
    client = None
    
    try:
        # Convert paths
        synonyms_path = Path(args.synonyms_file)
        schema_file = Path(__file__).parent / 'schema.sql'
        
        if not schema_file.exists():
            print(f"Error: Schema file not found: {schema_file}", file=sys.stderr)
            sys.exit(1)
        
        # Create use cases
        client, batch_import = create_use_cases(config, synonyms_path)
        
        # Recreate schema if requested
        if args.full:
            print("Recreating schema...")
            recreate_schema(client, schema_file)
            print("Schema recreated successfully")
        
        # Load resources
        print(f"Loading resources from {args.json_file}")
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        resources = data.get('resources', [])
        print(f"Total resources to process: {len(resources)}")
        
        # Import
        print(f"Starting import, incremental={args.incremental}")
        result = batch_import.execute(resources, incremental=args.incremental)
        
        print(f"Import completed: {result.to_dict()}")
        
        if result.error_count > 0:
            print(f"Warning: Failed to import {result.error_count} resources", file=sys.stderr)
            print(f"Check {args.error_log} for details", file=sys.stderr)
    
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if client:
            client.disconnect()


if __name__ == '__main__':
    main()