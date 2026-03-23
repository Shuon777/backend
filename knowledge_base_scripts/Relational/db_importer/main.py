import sys
import argparse
import logging
import traceback
import json
from pathlib import Path
from .config import DatabaseConfig
from .postgres_client import PostgresClient
from .schema_repository import PostgresSchemaRepository
from .importer import EcoAssistantImporter


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('db_importer.log', encoding='utf-8')
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='Database importer for eco_assistant schema')
    parser.add_argument('--full', action='store_true', help='Drop and recreate schema')
    parser.add_argument('--incremental', action='store_true', help='Import incrementally (skip duplicates)')
    parser.add_argument('--json-file', default='../../json_files/resources_dist.json', help='Path to JSON resources')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--error-log', default='import_errors.log', help='File to log errors')
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    error_logger = logging.getLogger('errors')
    error_handler = logging.FileHandler(args.error_log, encoding='utf-8')
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)

    config = DatabaseConfig.from_env()
    client = PostgresClient(config)
    
    try:
        client.connect()
        logger.info("Connected to database successfully")

        if args.full:
            logger.info("Starting full schema recreation")
            schema_file = Path(__file__).parent / 'schema.sql'
            if not schema_file.exists():
                logger.error(f"Schema file not found: {schema_file}")
                sys.exit(1)
            repo = PostgresSchemaRepository(client, schema_file)
            repo.drop_all()
            repo.create_all()
            logger.info("Schema recreated successfully")

        logger.info(f"Starting import from {args.json_file}")
        importer = EcoAssistantImporter(client)
        
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        resources = data.get('resources', [])
        
        logger.info(f"Total resources to process: {len(resources)}")
        
        result = importer.import_resources(args.json_file, incremental=args.incremental)
        
        logger.info(f"Import completed: {result}")
        
        if result['errors'] > 0:
            logger.warning(f"Failed to import {result['errors']} resources")
            logger.info(f"Check {args.error_log} for details")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        client.disconnect()
        logger.info("Disconnected from database")


if __name__ == '__main__':
    main()