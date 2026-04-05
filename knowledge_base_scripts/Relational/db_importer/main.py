import sys
import argparse
import json
from pathlib import Path

from .config import DatabaseConfig
from .adapters import (
    PostgresClient,
    PostgresResourceRepository,
    PostgresObjectRepository,
    PostgresObjectTypeRepository,
    PostgresSynonymRepository,
    PostgresModalityRepository,
    PostgresBibliographicRepository,
    PostgresCreationRepository,
    PostgresResourceStaticRepository,
    PostgresSupportMetadataRepository,
    PostgresSchemaRepository,
)
from .services import JsonSpeciesNormalizer, GeodataProvider
from .use_cases import ImportObjectsUseCase, ImportResourcesUseCase
from .infrastructure.logging_setup import setup_logging


def create_use_cases(config: DatabaseConfig, synonyms_path: Path, geodb_path: Path):
    client = PostgresClient(config)
    client.connect()

    resource_repo = PostgresResourceRepository(client)
    object_repo = PostgresObjectRepository(client)
    object_type_repo = PostgresObjectTypeRepository(client)
    synonym_repo = PostgresSynonymRepository(client)
    modality_repo = PostgresModalityRepository(client)
    bibliographic_repo = PostgresBibliographicRepository(client)
    creation_repo = PostgresCreationRepository(client)
    resource_static_repo = PostgresResourceStaticRepository(client)
    metadata_repo = PostgresSupportMetadataRepository(client)
    
    species_normalizer = JsonSpeciesNormalizer(synonyms_path)
    geodata_provider = GeodataProvider(geodb_path)
    
    import_objects = ImportObjectsUseCase(
        object_repo=object_repo,
        object_type_repo=object_type_repo,
        synonym_repo=synonym_repo
    )
    
    import_resources = ImportResourcesUseCase(
        resource_repo=resource_repo,
        object_repo=object_repo,
        resource_static_repo=resource_static_repo,
        metadata_repo=metadata_repo,
        bibliographic_repo=bibliographic_repo,
        creation_repo=creation_repo,
        modality_repo=modality_repo,
        geodata_provider=geodata_provider
    )
    
    return client, import_objects, import_resources


def recreate_schema(client: PostgresClient, schema_file: Path) -> None:
    repo = PostgresSchemaRepository(client, schema_file)
    repo.drop_all()
    repo.create_all()


def main() -> None:
    base_dir = Path('/var/www/salut_bot/json_files')
    
    parser = argparse.ArgumentParser(description='Database importer for eco_assistant schema')
    parser.add_argument('--full', action='store_true', help='Drop and recreate schema')
    parser.add_argument('--incremental', action='store_true', help='Import resources incrementally')
    parser.add_argument('--objects-file', default=str(base_dir / 'objects.json'), help='Path to objects.json')
    parser.add_argument('--resources-file', default=str(base_dir / 'resources.json'), help='Path to resources.json')
    parser.add_argument('--synonyms-file', default=str(base_dir / 'object_synonyms.json'), help='Path to synonyms file')
    parser.add_argument('--geodb-file', default=str(base_dir / 'geodb.json'), help='Path to geodb.json')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--error-log', default='import_errors.log', help='File to log errors')
    args = parser.parse_args()

    setup_logging(verbose=args.verbose, error_log=args.error_log)

    config = DatabaseConfig.from_env()
    client = None

    try:
        objects_path = Path(args.objects_file)
        resources_path = Path(args.resources_file)
        synonyms_path = Path(args.synonyms_file)
        geodb_path = Path(args.geodb_file)
        schema_file = Path(__file__).parent / 'schema.sql'

        if not schema_file.exists():
            print(f"Error: Schema file not found: {schema_file}", file=sys.stderr)
            sys.exit(1)

        client, import_objects, import_resources = create_use_cases(config, synonyms_path, geodb_path)

        if args.full:
            print("Recreating schema...")
            recreate_schema(client, schema_file)
            print("Schema recreated successfully")

        print(f"Loading objects from {objects_path}")
        if objects_path.exists():
            with open(objects_path, 'r', encoding='utf-8') as f:
                objects_data = json.load(f)
            objects_list = objects_data.get('objects', [])
            print(f"Total objects to process: {len(objects_list)}")
            
            objects_result = import_objects.execute(objects_list)
            print(f"Objects import: created={objects_result['created']}, updated={objects_result['updated']}, errors={objects_result['errors']}")
        else:
            print(f"Warning: Objects file not found: {objects_path}")
        
        print(f"Loading resources from {resources_path}")
        if resources_path.exists():
            with open(resources_path, 'r', encoding='utf-8') as f:
                resources_data = json.load(f)
            resources_list = resources_data.get('resources', [])
            print(f"Total resources to process: {len(resources_list)}")
            
            print(f"Starting import, incremental={args.incremental}")
            resources_result = import_resources.execute(resources_list, incremental=args.incremental)
            
            print(f"Resources import: success={resources_result['success']}, skipped={resources_result['skipped']}, errors={resources_result['errors']}")
            
            if resources_result['errors'] > 0:
                print(f"Warning: Failed to import {resources_result['errors']} resources", file=sys.stderr)
                print(f"Check {args.error_log} for details", file=sys.stderr)
        else:
            print(f"Warning: Resources file not found: {resources_path}")

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if client:
            client.disconnect()


if __name__ == '__main__':
    main()