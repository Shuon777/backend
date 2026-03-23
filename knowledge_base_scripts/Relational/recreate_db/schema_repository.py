import os
from pathlib import Path
from .interfaces import SchemaRepository, DatabaseClient

class PostgresSchemaRepository(SchemaRepository):
    def __init__(self, client: DatabaseClient, schema_file_path: Path):
        self._client = client
        self._schema_file_path = schema_file_path

    def drop_all(self) -> None:
        drop_script = """
        SET session_replication_role = replica;
        DROP SCHEMA IF EXISTS eco_assistant CASCADE;
        SET session_replication_role = DEFAULT;
        """
        self._client.execute_script(drop_script)

    def create_all(self) -> None:
        with open(self._schema_file_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        self._client.execute_script(schema_sql)