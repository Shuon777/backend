import logging
from pathlib import Path
from core.coordinates_finder import GeoProcessor
from core.relational_service import RelationalService
from core.search_service import SearchService
from infrastructure.db_utils_for_search import Slot_validator
from app.config import MAPS_DIR, DOMAIN, EMBEDDING_MODEL_PATH, FAISS_INDEX_PATH

logger = logging.getLogger(__name__)

geo = GeoProcessor(maps_dir=MAPS_DIR, domain=DOMAIN)
slot_val = Slot_validator()
search_service = SearchService(
    embedding_model_path=EMBEDDING_MODEL_PATH,
    faiss_index_path=FAISS_INDEX_PATH
)
relational_service = RelationalService()

logger.info("Services initialized")