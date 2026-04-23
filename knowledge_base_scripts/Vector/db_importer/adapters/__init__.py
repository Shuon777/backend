# adapters/__init__.py (обновленный)
from .json_resource_provider import JsonResourceProvider
from .text_extractor import NewResourceTextExtractor
from .chunker import FixedSizeChunker
from .embedding_service import HuggingFaceEmbeddingService
from .faiss_vector_store import FaissVectorStore

__all__ = [
    'JsonResourceProvider', 'NewResourceTextExtractor',
    'FixedSizeChunker', 'HuggingFaceEmbeddingService', 'FaissVectorStore'
]