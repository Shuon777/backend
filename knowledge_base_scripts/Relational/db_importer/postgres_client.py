import psycopg2
from psycopg2.extras import Json
from typing import Any, Dict, List, Optional, Tuple
from .config import DatabaseConfig
from .interfaces import DatabaseClient

class PostgresClient(DatabaseClient):
    def __init__(self, config: DatabaseConfig):
        self._config = config
        self._conn = None
        self._cur = None

    def connect(self) -> None:
        try:
            self._conn = psycopg2.connect(**self._config.__dict__)
            self._cur = self._conn.cursor()
        except Exception as e:
            raise RuntimeError(f"Failed to connect: {e}") from e

    def disconnect(self) -> None:
        if self._cur:
            self._cur.close()
        if self._conn:
            self._conn.close()

    def execute(self, sql: str, params: Optional[tuple] = None) -> None:
        if not self._cur:
            raise RuntimeError("Not connected")
        try:
            self._cur.execute(sql, params)
        except Exception as e:
            self._conn.rollback()
            raise RuntimeError(f"SQL error: {e}") from e

    def fetchone(self, sql: str, params: Optional[tuple] = None) -> Optional[tuple]:
        self.execute(sql, params)
        return self._cur.fetchone()

    def fetchall(self, sql: str, params: Optional[tuple] = None) -> List[tuple]:
        self.execute(sql, params)
        return self._cur.fetchall()

    def commit(self) -> None:
        if self._conn:
            self._conn.commit()

    def rollback(self) -> None:
        if self._conn:
            self._conn.rollback()