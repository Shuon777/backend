from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .entities import ResourceForIndexing, TextChunk


class ResourceProvider(ABC):
    @abstractmethod
    def get_resources(self) -> List[ResourceForIndexing]:
        pass


class TextExtractor(ABC):
    @abstractmethod
    def extract(self, resource: ResourceForIndexing) -> str:
        pass


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        pass


class EmbeddingService(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass


class VectorStoreService(ABC):
    @abstractmethod
    def add_documents(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass