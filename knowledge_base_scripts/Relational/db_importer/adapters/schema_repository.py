from pathlib import Path

from ..use_cases.interfaces import SchemaRepository
from .database_client import DatabaseClient


class PostgresSchemaRepository(SchemaRepository):
    
    def __init__(self, client: DatabaseClient, schema_file_path: Path):
        self._client = client
        self._schema_file_path = schema_file_path
    
    def drop_all(self) -> None:
        try:
            self._client.execute("DROP SCHEMA IF EXISTS eco_assistant CASCADE")
            self._client.commit()
        except Exception as e:
            self._client.rollback()
            raise RuntimeError(f"Failed to drop schema: {e}") from e
    
    def create_all(self) -> None:
        with open(self._schema_file_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        statements = self._split_sql_statements(schema_sql)
        
        for statement in statements:
            if statement.strip():
                try:
                    self._client.execute(statement)
                except Exception as e:
                    self._client.rollback()
                    raise RuntimeError(f"Failed to execute SQL: {statement[:200]}\nError: {e}") from e
        self._client.commit()
    
    def _split_sql_statements(self, sql: str) -> list:
        statements = []
        current = []
        
        for line in sql.split('\n'):
            stripped = line.strip()
            if stripped.startswith('--'):
                continue
            if stripped.startswith('CREATE EXTENSION') and 'IF NOT EXISTS' in stripped:
                continue
            current.append(line)
            if stripped.endswith(';'):
                statements.append('\n'.join(current))
                current = []
        
        if current:
            statements.append('\n'.join(current))
        
        return statements