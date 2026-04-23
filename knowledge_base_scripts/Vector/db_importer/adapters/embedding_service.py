from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..domain.interfaces import EmbeddingService


class HuggingFaceEmbeddingService(EmbeddingService):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self._model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_documents(texts)