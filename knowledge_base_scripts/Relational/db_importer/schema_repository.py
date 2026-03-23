from pathlib import Path
from .interfaces import DatabaseClient, SchemaRepository

class PostgresSchemaRepository(SchemaRepository):
    def __init__(self, client: DatabaseClient, schema_file_path: Path):
        self._client = client
        self._schema_file_path = schema_file_path

    def drop_all(self) -> None:
        drop_script = "DROP SCHEMA IF EXISTS eco_assistant CASCADE;"
        self._client.execute(drop_script)
        self._client.commit()

    def create_all(self) -> None:
        with open(self._schema_file_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        self._client.execute(schema_sql)
        self._client.commit()