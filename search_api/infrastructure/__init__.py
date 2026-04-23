# search_api/infrastructure/__init__.py (новый файл)
from .faiss_service import FaissService
from .redis_cache import RedisCache
from .database import init_db, get_session

__all__ = ['FaissService', 'RedisCache', 'init_db', 'get_session']