import psycopg2
from psycopg2.extensions import connection, cursor
from typing import Optional
from .config import DatabaseConfig
from .interfaces import DatabaseClient

class PostgresClient(DatabaseClient):
    def __init__(self, config: DatabaseConfig):
        self._config = config
        self._conn: Optional[connection] = None
        self._cur: Optional[cursor] = None

    def connect(self) -> None:
        try:
            self._conn = psycopg2.connect(**self._config.__dict__)
            self._cur = self._conn.cursor()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to database: {e}") from e

    def disconnect(self) -> None:
        if self._cur:
            self._cur.close()
        if self._conn:
            self._conn.close()

    def execute(self, sql: str) -> None:
        if not self._cur:
            raise RuntimeError("Not connected")
        try:
            self._cur.execute(sql)
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise RuntimeError(f"Failed to execute SQL: {e}") from e

    def execute_script(self, sql: str) -> None:
        self.execute(sql)