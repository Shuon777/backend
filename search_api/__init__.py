import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

for logger_name in ['search_api', 'search_api.adapters', 'search_api.use_cases', 'search_api.services', 'search_api.infrastructure', 'search_api.routes']:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)
    logging.getLogger(logger_name).propagate = True

logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.INFO)

from .routes.search import search_bp
from .config import SearchConfig
from .domain.entities import SearchRequest, SearchResponse, ObjectCriteria, ResourceCriteria
from .use_cases import SearchUseCase, SearchAndBuildUseCase
from .adapters.sqlalchemy_repository import SQLAlchemySearchRepository

__all__ = [
    'search_bp',
    'SearchConfig',
    'SearchRequest',
    'SearchResponse',
    'SearchUseCase',
    'SearchAndBuildUseCase',
    'ObjectCriteria',
    'ResourceCriteria',
    'SQLAlchemySearchRepository'
]