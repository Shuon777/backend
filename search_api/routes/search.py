import logging
from flask import request, jsonify, Blueprint, current_app

from ..config import SearchConfig
from ..use_cases import SearchUseCase, SearchAndBuildUseCase
from ..domain.entities import SearchRequest, ObjectCriteria, ResourceCriteria
from ..services import GeoMapService, LLMAnswerGenerator, ResponseBuilder
from ..infrastructure import RedisCache, init_db, get_session

logger = logging.getLogger(__name__)
search_bp = Blueprint('search_api', __name__)
logger.info("Importing SQLAlchemySearchRepository...")
from ..adapters.sqlalchemy_repository import SQLAlchemySearchRepository
logger.info("Import successful")

def _get_use_case():
    config = current_app.config.get('SEARCH_CONFIG')
    if not config:
        config = SearchConfig.from_env()
    
    cache = current_app.config.get('SEARCH_REDIS')
    if not cache:
        cache = RedisCache(config.redis_host, config.redis_port, config.redis_db)
    
    init_db(config)
    session_factory = get_session
    repository = SQLAlchemySearchRepository(session_factory)
    
    logger.error(f"=== REPOSITORY TYPE: {type(repository).__name__} ===")
    logger.error(f"=== REPOSITORY MODULE: {repository.__class__.__module__} ===")
    
    search_use_case = SearchUseCase(repository)
    geo_service = GeoMapService(config.maps_dir, config.domain)
    llm_generator = LLMAnswerGenerator()
    response_builder = ResponseBuilder(geo_service, llm_generator)
    return SearchAndBuildUseCase(search_use_case, response_builder, cache)


@search_bp.route('/search', methods=['POST'])
def search():
    logger.info("Search endpoint called")
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        if not data:
            data = {}

        sys_params = data.get('system_parameters', {})
        limit = sys_params.get('limit', data.get('limit', 20))
        offset = sys_params.get('offset', data.get('offset', 0))
        debug = sys_params.get('debug', data.get('debug', False))
        use_llm = sys_params.get('use_llm_answer', data.get('use_llm_answer', False))
        user_query = sys_params.get('user_query', data.get('user_query'))
        clean_user_query = sys_params.get('clean_user_query', data.get('clean_user_query'))

        search_params = data.get('search_parameters', data)

        object_criteria = None
        if search_params.get('object'):
            obj = search_params['object']
            object_criteria = ObjectCriteria(
                db_id=obj.get('identificator', {}).get('db_id') if obj.get('identificator') else None,
                name_synonyms=obj.get('name_synonyms'),
                properties=obj.get('properties'),
                object_type=obj.get('object_type')
            )
        logger.info(f"Object criteria properties: {object_criteria.properties if object_criteria else None}")
        
        resource_criteria = None
        if search_params.get('resource'):
            res = search_params['resource']
            features = None
            if res.get('features'):
                fd = res['features']
                if isinstance(fd, dict):
                    features = fd
                elif isinstance(fd, list):
                    features = {f['name']: f['value'] for f in fd if isinstance(f, dict)}
            resource_criteria = ResourceCriteria(
                title=res.get('title'),
                uri=res.get('identificator', {}).get('uri') if res.get('identificator') else None,
                author=res.get('bibliographic', {}).get('author') if res.get('bibliographic') else None,
                source=res.get('bibliographic', {}).get('source') if res.get('bibliographic') else None,
                modality_type=search_params.get('modality_type') or res.get('modality', {}).get('type'),
                features=features,
                structured_data=res.get('modality', {}).get('value', {}).get('structured_data') if res.get('modality') else None,
                taxonomy=res.get('modality', {}).get('value', {}).get('structured_data', {}).get('taxonomy') if res.get('modality') else None
            )

        request_obj = SearchRequest(
            object=object_criteria,
            resource=resource_criteria,
            modality_type=search_params.get('modality_type'),
            limit=limit,
            offset=offset,
            debug=debug,
            use_llm_answer=use_llm,
            user_query=user_query,
            clean_user_query=clean_user_query
        )

        logger.info("Creating use case...")
        use_case = _get_use_case()
        logger.info("Use case created, executing...")
        result = use_case.execute(request_obj)
        logger.info("Search completed successfully")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500