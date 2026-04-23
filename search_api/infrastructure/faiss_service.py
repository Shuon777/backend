# search_api/infrastructure/faiss_service.py
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ..config import SearchConfig

logger = logging.getLogger(__name__)


class FaissService:
    def __init__(self, config: SearchConfig):
        self.config = config
        self._vectorstore: Optional[FAISS] = None
        self._embedding_model: Optional[HuggingFaceEmbeddings] = None
        self._resources_by_id: Dict[str, Dict] = {}
        self._load_resources_data()
        self._init_embedding_model()

    def _get_embedding_model_path(self) -> str:
        project_root = Path(__file__).parent.parent.parent
        active_model_file = project_root / "embedding_models" / "active_model.json"
        
        if active_model_file.exists():
            try:
                with open(active_model_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    model_path = data.get('model_path')
                    if model_path and Path(model_path).exists():
                        return str(model_path)
                    active_model = data.get('active_model')
                    if active_model:
                        full_path = project_root / "embedding_models" / active_model
                        if full_path.exists():
                            return str(full_path)
                        full_path = project_root / "embedding_models" / active_model.split('/')[-1]
                        if full_path.exists():
                            return str(full_path)
                    logger.warning(f"Model path from active_model.json not found: {model_path or active_model}")
            except Exception as e:
                logger.warning(f"Failed to read active_model.json: {e}")
        
        if self.config.embedding_model_path and Path(self.config.embedding_model_path).exists():
            return self.config.embedding_model_path
        
        fallback_path = project_root / "embedding_models" / "bge-m3"
        if fallback_path.exists():
            return str(fallback_path)
        
        fallback_path = project_root / "embedding_models" / "sbert_large_nlu_ru"
        if fallback_path.exists():
            return str(fallback_path)
        
        raise FileNotFoundError("No embedding model found")
    def _init_embedding_model(self):
        model_path = self._get_embedding_model_path()
        logger.info(f"Loading embedding model from: {model_path}")
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Embedding model loaded successfully")

    def _load_resources_data(self):
        base_dir = Path(__file__).parent.parent.parent
        resources_file = base_dir / "json_files" / "resources_dist.json"
        if not resources_file.exists():
            logger.warning(f"FAISS resources file not found: {resources_file}")
            return
        try:
            with open(resources_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for res in data.get('resources', []):
                rid = res.get('identificator', {}).get('id')
                if rid:
                    self._resources_by_id[str(rid)] = res
            logger.info(f"Loaded {len(self._resources_by_id)} resources for FAISS enrichment")
        except Exception as e:
            logger.error(f"Failed to load resources_dist.json: {e}")

    def load_index(self) -> Optional[FAISS]:
        if self._vectorstore is not None:
            return self._vectorstore
        index_path = getattr(self.config, 'faiss_index_path', None)
        if not index_path:
            logger.warning("FAISS index path not configured")
            return None
        try:
            self._vectorstore = FAISS.load_local(
                index_path,
                self._embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS index loaded from {index_path}, vectors: {self._vectorstore.index.ntotal}")
            return self._vectorstore
        except Exception as e:
            logger.error(f"FAISS load error: {e}")
            return None

    def search(self, query: str, k: int = 20, similarity_threshold: float = 0.03) -> List[Dict[str, Any]]:
        vs = self.load_index()
        if vs is None:
            return []
        try:
            results = vs.similarity_search_with_score(query, k=k)
            filtered = []
            for doc, score in results:
                if score >= similarity_threshold:
                    resource_id = doc.metadata.get('resource_id')
                    full_doc = self._get_full_document(resource_id, doc.page_content)
                    filtered.append({
                        'content': full_doc,
                        'similarity': float(score),
                        'source': 'faiss_vector_search',
                        'object_name': doc.metadata.get('common_name', ''),
                        'object_type': doc.metadata.get('resource_type', 'unknown'),
                        'feature_data': {
                            'in_stoplist': doc.metadata.get('in_stoplist', 1),
                            'source': doc.metadata.get('source', '')
                        },
                        'resource_id': resource_id,
                        'title': doc.metadata.get('title', '')
                    })
            filtered.sort(key=lambda x: x['similarity'], reverse=True)
            logger.info(f"FAISS search returned {len(filtered)} results (threshold={similarity_threshold})")
            return filtered
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []

    def _get_full_document(self, resource_id: str, chunk_content: str) -> str:
        if not resource_id or resource_id not in self._resources_by_id:
            return chunk_content
        res = self._resources_by_id[resource_id]
        res_type = res.get('type')
        if res_type == 'Текст':
            content = res.get('content', '')
            if content:
                return content
            structured = res.get('structured_data', {})
            if structured:
                return self._structured_to_text(structured)
        elif res_type == 'Географический объект':
            common = res.get('identificator', {}).get('name', {}).get('common', '')
            desc = res.get('description', '')
            return f"{common}\n{desc}".strip()
        return chunk_content

    def _structured_to_text(self, data: Dict) -> str:
        parts = []
        for section, content in data.items():
            if isinstance(content, dict):
                section_text = f"{section}:"
                for k, v in content.items():
                    if v and str(v).strip() and v != '-':
                        section_text += f"\n  {k}: {v}"
                parts.append(section_text)
        return "\n\n".join(parts)