import pytest
import json

class TestSmokeTests:
    def test_health_endpoint(self, client):
        response = client.get('/')
        assert response.status_code == 200
        assert b'SalutBot API works' in response.data

    def test_search_endpoint_responds(self, client):
        response = client.post('/search', json={})
        assert response.status_code == 200

    def test_search_with_minimal_data(self, client):
        request_data = {"limit": 5, "offset": 0}
        response = client.post('/search', json=request_data)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'objects' in data
        assert 'resources' in data

    def test_search_response_structure(self, client):
        request_data = {
            "object": {"object_type": "Test"},
            "resource": {"title": "Test"}
        }
        response = client.post('/search', json=request_data)
        assert response.status_code == 200
        data = json.loads(response.data)
        expected_keys = ['object_criteria', 'resource_criteria', 'modality_filter', 'objects', 'resources']
        for key in expected_keys:
            assert key in data

    def test_search_with_debug_mode(self, client):
        request_data = {
            "system_parameters": {"debug": True},
            "object": {"object_type": "Plant"}
        }
        response = client.post('/search', json=request_data)
        assert response.status_code == 200
        data = json.loads(response.data)
        if data.get('debug'):
            assert 'objects_query_time' in data['debug'] or 'search_time' in data['debug']

    def test_database_connection(self, db_client):
        result = db_client.fetchone("SELECT 1 as test")
        assert result is not None
        assert result[0] == 1