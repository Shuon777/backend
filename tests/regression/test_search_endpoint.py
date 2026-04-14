# tests/regression/test_search_endpoint.py
import pytest

pytestmark = pytest.mark.regression

class TestSearchEndpointRegression:
    def test_museums_search_first_6_results(self, production_client):
        request_body = {
            "system_parameters": {
                "user_query": "Сколько музеев?",
                "use_llm_answer": False,
                "limit": 6,
                "offset": 0
            },
            "search_parameters": {
                "object": {
                    "properties": {
                        "subtypes": "Музеи"
                    }
                }
            }
        }
        
        response = production_client.post('/search', json=request_body)
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert 'objects' in data
        assert len(data['objects']) == 6
        
        expected_ids = [6785, 6753, 6756, 6757, 6758, 6759]
        actual_ids = [obj['id'] for obj in data['objects']]
        
        assert actual_ids == expected_ids