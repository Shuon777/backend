# search_api/routes/search.py
from flask import request, jsonify, Blueprint
from ..config import SearchConfig
from ..adapters.database import PostgresSearchRepository
from ..use_cases.search_service import SearchUseCase
from ..domain.entities import SearchRequest, ObjectCriteria, ResourceCriteria

search_bp = Blueprint('search_api', __name__)

def _search_use_case():
    config = SearchConfig.from_env()
    repository = PostgresSearchRepository(config)
    return SearchUseCase(repository)

@search_bp.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    sys_params = data.get('system_parameters', {})
    limit = sys_params.get('limit', data.get('limit', 20))
    offset = sys_params.get('offset', data.get('offset', 0))
    debug = sys_params.get('debug', data.get('debug', False))
    search_type = sys_params.get('search_type', data.get('search_type', 'Relational'))
    use_llm_answer = sys_params.get('use_llm_answer', data.get('use_llm_answer', False))
    user_query = sys_params.get('user_query', data.get('user_query'))
    clean_user_query = sys_params.get('clean_user_query', data.get('clean_user_query'))
    
    search_params = data.get('search_parameters', data)
    
    object_criteria = None
    if search_params.get('object'):
        obj_data = search_params['object']
        object_criteria = ObjectCriteria(
            db_id=obj_data.get('identificator', {}).get('db_id') if obj_data.get('identificator') else None,
            name_synonyms=obj_data.get('name_synonyms'),
            properties=obj_data.get('properties'),
            object_type=obj_data.get('object_type')
        )
    
    resource_criteria = None
    if search_params.get('resource'):
        res_data = search_params['resource']
        features = None
        if res_data.get('features'):
            features_data = res_data['features']
            if isinstance(features_data, dict):
                features = features_data
            elif isinstance(features_data, list):
                features = {f['name']: f['value'] for f in features_data if isinstance(f, dict)}
        resource_criteria = ResourceCriteria(
            title=res_data.get('title'),
            uri=res_data.get('identificator', {}).get('uri') if res_data.get('identificator') else None,
            author=res_data.get('bibliographic', {}).get('author') if res_data.get('bibliographic') else None,
            source=res_data.get('bibliographic', {}).get('source') if res_data.get('bibliographic') else None,
            modality_type=search_params.get('modality_type') or res_data.get('modality', {}).get('type'),
            features=features,
            structured_data=res_data.get('modality', {}).get('value', {}).get('structured_data') if res_data.get('modality') else None,
            taxonomy=res_data.get('modality', {}).get('value', {}).get('structured_data', {}).get('taxonomy') if res_data.get('modality') else None
        )
    
    request_obj = SearchRequest(
        object=object_criteria,
        resource=resource_criteria,
        modality_type=search_params.get('modality_type'),
        limit=limit,
        offset=offset,
        debug=debug,
        search_type=search_type,
        use_llm_answer=use_llm_answer,
        user_query=user_query,
        clean_user_query=clean_user_query
    )
    
    use_case = _search_use_case()
    response = use_case.execute(request_obj)
    
    result = {
        'object_criteria': {
            'db_id': response.object_criteria.db_id if response.object_criteria else None,
            'name_synonyms': response.object_criteria.name_synonyms if response.object_criteria else None,
            'properties': response.object_criteria.properties if response.object_criteria else None,
            'object_type': response.object_criteria.object_type if response.object_criteria else None
        } if response.object_criteria else None,
        'resource_criteria': {
            'title': response.resource_criteria.title if response.resource_criteria else None,
            'author': response.resource_criteria.author if response.resource_criteria else None,
            'source': response.resource_criteria.source if response.resource_criteria else None,
            'modality_type': response.resource_criteria.modality_type if response.resource_criteria else None,
            'features': response.resource_criteria.features if response.resource_criteria else None
        } if response.resource_criteria else None,
        'modality_filter': response.modality_filter,
        'objects': [
            {
                'id': o.id,
                'db_id': o.db_id,
                'type': o.object_type,
                'properties': o.properties,
                'synonyms': o.synonyms
            } for o in response.objects
        ],
        'resources': [
            {
                'id': r.id,
                'title': r.title,
                'uri': r.uri,
                'author': r.author,
                'source': r.source,
                'modality_type': r.modality_type,
                'features': r.features,
                'content': r.content
            } for r in response.resources
        ]
    }
    if response.debug_info:
        result['debug'] = response.debug_info
    if response.llm_answer:
        result['llm_answer'] = response.llm_answer
    return jsonify(result), 200