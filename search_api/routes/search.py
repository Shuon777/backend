# search_api/routes/search.py
from flask import request, jsonify, Blueprint
from ..config import SearchConfig
from ..adapters.database import PostgresSearchRepository
from ..use_cases.search_service import SearchUseCase
from ..domain.entities import SearchRequest

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
    
    query = data.get('query')
    if not query:
        return jsonify({'error': 'Field "query" is required'}), 400
    
    request_obj = SearchRequest(
        query=query,
        modality_type=data.get('modality_type')
    )
    
    use_case = _search_use_case()
    response = use_case.execute(request_obj)
    
    return jsonify({
        'query': response.query,
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
                'modality_type': r.modality_type,
                'content': r.content
            } for r in response.resources
        ]
    }), 200