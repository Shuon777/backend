from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangDocument

from ..domain.entities import TextChunk
from ..domain.interfaces import VectorStoreService, EmbeddingService


class FaissVectorStore(VectorStoreService):
    def __init__(self, embedding_service: EmbeddingService):
        self._embedding_service = embedding_service
        self._documents = []
        self._embeddings = []

    def add_documents(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> None:
        self._documents.extend(chunks)
        self._embeddings.extend(embeddings)

    def save(self, path: str) -> None:
        if not self._documents:
            raise RuntimeError("No documents to save")

        Path(path).mkdir(parents=True, exist_ok=True)

        docs = [
            LangDocument(page_content=c.text, metadata=c.metadata)
            for c in self._documents
        ]
        vectorstore = FAISS.from_documents(docs, self._embedding_service._model)
        vectorstore.save_local(path)