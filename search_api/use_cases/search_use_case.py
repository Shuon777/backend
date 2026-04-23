import time
import logging
from dataclasses import dataclass
from typing import List, Optional

from ..domain.entities import SearchRequest, SearchResponse, ResourceCriteria, ObjectResult, ResourceResult
from ..adapters.search_repository import SearchRepository
from ..infrastructure.faiss_service import FaissService

logger = logging.getLogger(__name__)

@dataclass
class SearchUseCase:
    _repository: SearchRepository
    faiss_service: Optional[FaissService] = None

    def execute(self, request: SearchRequest) -> SearchResponse:
        start_time = time.time()
        debug = {}
        objects: List[ObjectResult] = []
        object_ids: Optional[List[int]] = None
        resources: List[ResourceResult] = []

        query_for_search = request.clean_user_query if request.clean_user_query else request.user_query

        logger.info(f"SearchUseCase.execute START")
        logger.info(f"force_vector_search={request.force_vector_search}, use_vector_fallback={request.use_vector_fallback}")
        logger.info(f"query_for_search={query_for_search[:100] if query_for_search else None}")

        if request.object:
            obj_start = time.time()
            objects = self._repository.find_objects_by_criteria(
                request.object, limit=request.limit, offset=request.offset
            )
            debug['objects_query_time'] = time.time() - obj_start
            object_ids = [obj.id for obj in objects] if objects else None
            logger.info(f"Found {len(objects)} objects")

        use_faiss = False
        if request.force_vector_search and query_for_search:
            use_faiss = True
            debug['faiss_reason'] = 'force_vector_search'
            logger.info("FAISS forced by force_vector_search parameter")
        elif request.use_vector_fallback and query_for_search:
            if request.object and not objects:
                use_faiss = True
                debug['faiss_reason'] = 'no_relational_objects'
                logger.info("FAISS fallback: no relational objects found")
            elif not request.object:
                use_faiss = True
                debug['faiss_reason'] = 'no_object_criteria'
                logger.info("FAISS fallback: no object criteria provided")

        if use_faiss and self.faiss_service and query_for_search:
            faiss_start = time.time()
            logger.info(f"Executing FAISS search with query: {query_for_search[:100]}...")
            faiss_results = self.faiss_service.search(
                query=query_for_search,
                k=request.limit,
                similarity_threshold=request.vector_similarity_threshold
            )
            debug['faiss_query_time'] = time.time() - faiss_start
            debug['faiss_results_count'] = len(faiss_results)
            logger.info(f"FAISS search returned {len(faiss_results)} results")
            resources = [self._faiss_result_to_resource(r) for r in faiss_results]
        else:
            if not use_faiss:
                logger.info("FAISS not used, proceeding with relational search")
            elif not self.faiss_service:
                logger.warning("FAISS service not available")
            elif not query_for_search:
                logger.warning("FAISS skipped: no query provided")
            
            if request.object and not objects and not use_faiss:
                debug['resources_skipped'] = True
                logger.info("Skipping resource search because no objects found and FAISS not used")
            else:
                res_start = time.time()
                resources = self._repository.find_resources_by_criteria(
                    request.resource or ResourceCriteria(),
                    object_ids,
                    limit=request.limit,
                    offset=request.offset
                )
                debug['resources_query_time'] = time.time() - res_start
                logger.info(f"Relational search returned {len(resources)} resources")

        debug['total_time'] = time.time() - start_time
        logger.info(f"SearchUseCase.execute END, total_time: {debug['total_time']:.3f}s")

        return SearchResponse(
            object_criteria=request.object,
            resource_criteria=request.resource,
            modality_filter=request.modality_type,
            objects=objects,
            resources=resources,
            debug_info=debug if request.debug else None
        )

    def _faiss_result_to_resource(self, fr: dict) -> ResourceResult:
        return ResourceResult(
            id=hash(fr.get('resource_id', str(fr.get('similarity', 0)))),
            title=fr.get('title', ''),
            uri=None,
            author=None,
            source=None,
            modality_type='Текст',
            content={'structured_data': fr.get('content', '')},
            features=fr.get('feature_data', {})
        )