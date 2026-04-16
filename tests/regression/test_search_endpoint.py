# tests/regression/test_search_endpoint.py
import pytest

pytestmark = pytest.mark.regression

class TestSearchEndpointRegression:
    def test_museums_search_first_6_results(self, production_client):
        """TC-01: Поиск музеев - проверка первых 6 объектов"""
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

    def test_bodaybo_museum_search(self, production_client):
        """TC-02: Поиск музеев в Бодайбо с текстовыми ресурсами"""
        request_body = {
            "system_parameters": {
                "user_query": "Расскажи о музеях в Бодайбо",
                "use_llm_answer": False,
                "limit": 10,
                "offset": 0
            },
            "search_parameters": {
                "object": {
                    "properties": {
                        "subtypes": "Музеи",
                        "exact_location": "город Бодайбо, Иркутская область, Россия"
                    }
                },
                "resource": {
                    "modality": {
                        "type": "Текст"
                    }
                },
                "modality_type": "Текст"
            }
        }
        
        response = production_client.post('/search', json=request_body)
        
        assert response.status_code == 200
        data = response.get_json()
        
        museum = data['objects'][0]
        assert museum['id'] == 6899, "ID объекта должен быть 6899"
        assert museum['db_id'] == "GEO_OBJ_123acf0dbc28"

        resource = data['resources'][0]
        assert resource['modality_type'] == "Текст"
        assert resource['id'] == 16959, "ID ресурса должен быть 16959"
        assert resource['title'] == "Бодайбинский городской краеведческий музей имени В. Ф. Верещагина"
        assert resource['source'] == "Байкальский музей СО РАН"
        
        content = resource.get('content', {})
        structured_data = content.get('structured_data', {})
        assert 'content' in structured_data
        assert "Краеведческий музей" in structured_data['content']

        assert 'debug' in data, "Debug режим должен быть включен"