import sys
from pathlib import Path
from .config import DatabaseConfig
from .postgres_client import PostgresClient
from .schema_repository import PostgresSchemaRepository
from .recreate_db_usecase import RecreateDatabaseUseCase

def main():
    config = DatabaseConfig.from_env()
    client = PostgresClient(config)
    try:
        client.connect()
        schema_file = Path(__file__).parent / 'schema.sql'
        repo = PostgresSchemaRepository(client, schema_file)
        usecase = RecreateDatabaseUseCase(repo)
        usecase.execute()
        print("Database schema recreated successfully.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.disconnect()

if __name__ == '__main__':
    main()