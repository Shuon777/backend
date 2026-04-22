# tests/regression/test_search_endpoint.py (исправленный)

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
        
        for obj in data['objects']:
            props = obj.get('properties', {})
            subtypes = props.get('subtypes')
            if isinstance(subtypes, str):
                subtypes = [subtypes]
            assert "Музеи" in subtypes, f"Object {obj.get('id')} does not have 'Музеи' in subtypes"
        
        museums_count = len(data['objects'])
        assert museums_count == 6

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
        assert len(data['objects']) > 0
        
        museum = data['objects'][0]
        assert 'db_id' in museum
        assert 'properties' in museum
        
        assert len(data['resources']) > 0
        resource = data['resources'][0]
        assert resource['modality_type'] == "Текст"
        assert 'content' in resource
        
        if resource.get('content'):
            content = resource['content']
            if isinstance(content, dict) and 'structured_data' in content:
                structured_data = content['structured_data']
                assert 'content' in structured_data

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
        
        has_baikal = False
        for obj in data['objects']:
            props = obj.get('properties', {})
            subtypes = props.get('subtypes')
            if isinstance(subtypes, str):
                subtypes = [subtypes]
            assert "Наука" in subtypes
            
            location = props.get('exact_location', '')
            name = props.get('name', '')
            combined = (str(location) + ' ' + str(name)).lower()
            if 'байкал' in combined:
                has_baikal = True
                break
        
        if not has_baikal:
            for obj in data['objects']:
                props = obj.get('properties', {})
                combined = str(props).lower()
                if 'байкал' in combined:
                    has_baikal = True
                    break
        
        assert has_baikal