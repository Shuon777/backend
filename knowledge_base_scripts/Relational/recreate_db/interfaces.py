from abc import ABC, abstractmethod

class DatabaseClient(ABC):
    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def execute(self, sql: str) -> None:
        pass

    @abstractmethod
    def execute_script(self, sql: str) -> None:
        pass

class SchemaRepository(ABC):
    @abstractmethod
    def drop_all(self) -> None:
        pass

    @abstractmethod
    def create_all(self) -> None:
        pass