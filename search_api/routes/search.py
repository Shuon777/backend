from flask import request, jsonify, Blueprint, current_app

from ..config import SearchConfig
from ..adapters.database import PostgresSearchRepository
from ..use_cases import SearchUseCase, SearchAndBuildUseCase
from ..domain.entities import SearchRequest, ObjectCriteria, ResourceCriteria
from ..services import GeoMapService, LLMAnswerGenerator, ResponseBuilder
from ..infrastructure import RedisCache

search_bp = Blueprint('search_api', __name__)


def _get_use_case():
    config = current_app.config.get('SEARCH_CONFIG')
    if not config:
        config = SearchConfig.from_env()
    
    cache = current_app.config.get('SEARCH_REDIS')
    if not cache:
        cache = RedisCache(config.redis_host, config.redis_port, config.redis_db)
    
    repository = PostgresSearchRepository(config)
    search_use_case = SearchUseCase(repository)
    geo_service = GeoMapService(config.maps_dir, config.domain)
    llm_generator = LLMAnswerGenerator()
    response_builder = ResponseBuilder(geo_service, llm_generator)
    return SearchAndBuildUseCase(search_use_case, response_builder, cache)


@search_bp.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

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

    use_case = _get_use_case()
    result = use_case.execute(request_obj)
    return jsonify(result), 200