import logging
import time
from dataclasses import dataclass

from ..domain.entities import SearchRequest
from ..infrastructure.redis_cache import RedisCache
from ..services.response_builder import ResponseBuilder
from .search_use_case import SearchUseCase

logger = logging.getLogger(__name__)

@dataclass
class SearchAndBuildUseCase:
    _search_use_case: SearchUseCase
    _response_builder: ResponseBuilder
    _cache: RedisCache
    _cache_ttl: int = 3600

    def execute(self, request: SearchRequest) -> dict:
        if not request.debug:
            cache_key = RedisCache.generate_key('search_response', self._cache_params(request))
            cache_hit, cached = self._cache.get(cache_key)
            if cache_hit:
                logger.info("=== CACHE HIT, returning cached result ===")
                return cached
            logger.info("=== CACHE MISS, executing search ===")
        else:
            logger.info("=== DEBUG MODE, skipping cache ===")

        start_time = time.time()
        search_response = self._search_use_case.execute(request)
        
        if search_response.debug_info is None:
            search_response.debug_info = {}
        
        search_response.debug_info['search_time'] = time.time() - start_time

        result = self._response_builder.build(
            search_response,
            user_query=request.user_query,
            use_llm=request.use_llm_answer
        )
        
        if request.debug:
            result['debug'] = search_response.debug_info
        elif not request.debug:
            self._cache.set(cache_key, result, expire_seconds=self._cache_ttl)
        
        return result

    def _cache_params(self, request: SearchRequest) -> dict:
        return {
            'object': request.object.to_dict() if request.object else None,
            'resource': request.resource.to_dict() if request.resource else None,
            'modality_type': request.modality_type,
            'limit': request.limit,
            'offset': request.offset,
            'use_llm': request.use_llm_answer,
            'user_query': request.user_query,
            'clean_user_query': request.clean_user_query,
        }