"""Database client interface and PostgreSQL implementation."""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import psycopg2


class DatabaseClient(ABC):
    """Interface for database client."""
    
    @abstractmethod
    def connect(self) -> None:
        """Establish database connection."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    def execute(self, sql: str, params: Optional[tuple] = None) -> None:
        """Execute SQL statement."""
        pass
    
    @abstractmethod
    def fetchone(self, sql: str, params: Optional[tuple] = None) -> Optional[tuple]:
        """Fetch single row."""
        pass
    
    @abstractmethod
    def fetchall(self, sql: str, params: Optional[tuple] = None) -> List[tuple]:
        """Fetch all rows."""
        pass
    
    @abstractmethod
    def commit(self) -> None:
        """Commit transaction."""
        pass
    
    @abstractmethod
    def rollback(self) -> None:
        """Rollback transaction."""
        pass


class PostgresClient(DatabaseClient):
    """PostgreSQL database client implementation."""
    
    def __init__(self, config):
        self._config = config
        self._conn = None
        self._cur = None
    
    def connect(self) -> None:
        """Establish PostgreSQL connection."""
        try:
            self._conn = psycopg2.connect(**self._config.__dict__)
            self._cur = self._conn.cursor()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to database: {e}") from e
    
    def disconnect(self) -> None:
        """Close PostgreSQL connection."""
        if self._cur:
            self._cur.close()
        if self._conn:
            self._conn.close()
    
    def execute(self, sql: str, params: Optional[tuple] = None) -> None:
        """Execute SQL statement."""
        if not self._cur:
            raise RuntimeError("Not connected to database")
        try:
            self._cur.execute(sql, params)
        except Exception as e:
            self._conn.rollback()
            raise RuntimeError(f"SQL execution error: {e}") from e
    
    def fetchone(self, sql: str, params: Optional[tuple] = None) -> Optional[tuple]:
        """Fetch single row."""
        self.execute(sql, params)
        return self._cur.fetchone()
    
    def fetchall(self, sql: str, params: Optional[tuple] = None) -> List[tuple]:
        """Fetch all rows."""
        self.execute(sql, params)
        return self._cur.fetchall()
    
    def commit(self) -> None:
        """Commit transaction."""
        if self._conn:
            self._conn.commit()
    
    def rollback(self) -> None:
        """Rollback transaction."""
        if self._conn:
            self._conn.rollback()