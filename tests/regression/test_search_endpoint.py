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
                "offset": 0,
                "debug": True
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
        
        assert 'debug' in data
        assert 'objects' in data
        assert len(data['objects']) == 6
        
        expected_ids = [6785, 6753, 6756, 6757, 6758, 6759]
        actual_ids = [obj['id'] for obj in data['objects']]
        
        assert actual_ids == expected_ids

    def test_bodaybo_museum_search(self, production_client):
        request_body = {
            "system_parameters": {
                "user_query": "Расскажи о музеях в Бодайбо",
                "use_llm_answer": False,
                "limit": 6,
                "offset": 0,
                "debug": True
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
        
        assert 'debug' in data
        
        museum = data['objects'][0]
        assert museum['id'] == 6899
        assert museum['db_id'] == "GEO_OBJ_123acf0dbc28"

        resource = data['resources'][0]
        assert resource['modality_type'] == "Текст"
        assert resource['id'] == 16959
        assert resource['title'] == "Бодайбинский городской краеведческий музей имени В. Ф. Верещагина"
        assert resource['source'] == "Байкальский музей СО РАН"
        
        content = resource.get('content', {})
        structured_data = content.get('structured_data', {})
        assert 'content' in structured_data
        assert "Краеведческий музей" in structured_data['content']

    def test_scientific_institutions_near_baikal(self, production_client):
        request_body = {
            "system_parameters": {
                "user_query": "Какие научные учреждения есть около Байкала?",
                "use_llm_answer": False,
                "limit": 6,
                "offset": 0,
                "debug": True
            },
            "search_parameters": {
                "object": {
                    "properties": {
                        "subtypes": "Наука"
                    }
                },
                "modality_type": "Текст"
            }
        }
        
        response = production_client.post('/search', json=request_body)
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert 'debug' in data
        assert 'objects' in data
        assert len(data['objects']) > 0
        
        for obj in data['objects']:
            props = obj.get('properties', {})
            subtypes = props.get('subtypes')
            if isinstance(subtypes, str):
                subtypes = [subtypes]
            assert subtypes is not None
            assert "Наука" in subtypes
        
        has_baikal = False
        for obj in data['objects']:
            props = obj.get('properties', {})
            location = props.get('exact_location', '')
            name = props.get('name', '')
            combined = (location + ' ' + name).lower()
            if 'байкал' in combined:
                has_baikal = True
                break
        assert has_baikal