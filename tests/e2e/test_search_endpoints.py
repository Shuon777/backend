import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

class TestSearchEndpointsE2E:
    def test_search_by_chabrets(self, client, db_client):
        request_data = {
            "object": {
                "name_synonyms": {"ru_names": ["чабрец"]}
            },
            "resource": {"features": {"season": "Весна"}},
            "modality_type": "Изображение"
        }
        
        response = client.post('/search', json=request_data)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'objects' in data
        assert 'resources' in data

    def test_search_by_author_ivanov(self, client, db_client):
        request_data = {"resource": {"bibliographic": {"author": "Иванов"}}}
        response = client.post('/search', json=request_data)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'resources' in data

    def test_search_by_title_baikal(self, client, db_client):
        request_data = {"resource": {"title": "Байкал"}}
        response = client.post('/search', json=request_data)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_search_by_nerpa_and_taxonomy(self, client, db_client):
        request_data = {
            "object": {"name_synonyms": {"ru_names": ["нерпа"]}},
            "resource": {"taxonomy": {"family": "Phocidae"}},
            "modality_type": "Текст"
        }
        response = client.post('/search', json=request_data)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'modality_filter' in data

    def test_search_by_angara_geodata(self, client, db_client):
        request_data = {
            "object": {"name_synonyms": {"ru_names": ["ангара"]}},
            "modality_type": "Геоданные"
        }
        response = client.post('/search', json=request_data)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data.get('modality_filter') == 'Геоданные'

    def test_search_invalid_json(self, client):
        response = client.post('/search', data='invalid json', content_type='application/json')
        assert response.status_code == 400

    def test_search_empty_request(self, client):
        response = client.post('/search', json={})
        assert response.status_code == 200