import pytest
from search_api.domain.entities import ObjectCriteria, ResourceCriteria

class TestPostgresSearchRepository:
    def test_find_objects_by_criteria_with_db_id(self, search_repository, db_client):
        db_client.execute("""
            INSERT INTO eco_assistant.object_type (name, schema) 
            VALUES ('TestType', '{}') ON CONFLICT (name) DO NOTHING
        """)
        db_client.commit()
        row = db_client.fetchone("SELECT id FROM eco_assistant.object_type WHERE name = 'TestType'")
        type_id = row[0]
        db_client.execute("""
            INSERT INTO eco_assistant.object (db_id, object_type_id, object_properties)
            VALUES ('test_db_001', %s, '{}')
        """, (type_id,))
        db_client.commit()
        criteria = ObjectCriteria(db_id='test_db_001')
        results = search_repository.find_objects_by_criteria(criteria)
        assert len(results) >= 1
        assert any(r.db_id == 'test_db_001' for r in results)

    def test_find_objects_by_criteria_with_type(self, search_repository, db_client):
        db_client.execute("""
            INSERT INTO eco_assistant.object_type (name, schema) 
            VALUES ('Flora', '{}') ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        """)
        db_client.commit()
        row = db_client.fetchone("SELECT id FROM eco_assistant.object_type WHERE name = 'Flora'")
        type_id = row[0]
        db_client.execute("""
            INSERT INTO eco_assistant.object (db_id, object_type_id, object_properties)
            VALUES ('flora_001', %s, '{}')
        """, (type_id,))
        db_client.commit()
        criteria = ObjectCriteria(object_type='Flora')
        results = search_repository.find_objects_by_criteria(criteria)
        assert len(results) >= 1
        assert any(r.object_type == 'Flora' for r in results)

    def test_find_resources_by_criteria_with_title(self, search_repository, db_client):
        db_client.execute("INSERT INTO eco_assistant.bibliographic DEFAULT VALUES RETURNING id")
        bib_id = db_client.fetchone("SELECT lastval()")[0]
        db_client.execute("INSERT INTO eco_assistant.creation DEFAULT VALUES RETURNING id")
        crea_id = db_client.fetchone("SELECT lastval()")[0]
        db_client.execute("""
            INSERT INTO eco_assistant.resource_static (bibliographic_id, creation_id)
            VALUES (%s, %s) RETURNING id
        """, (bib_id, crea_id))
        static_id = db_client.fetchone("SELECT lastval()")[0]
        db_client.execute("""
            INSERT INTO eco_assistant.support_metadata (parameters) VALUES ('{}') RETURNING id
        """)
        meta_id = db_client.fetchone("SELECT lastval()")[0]
        db_client.execute("""
            INSERT INTO eco_assistant.resource (title, resource_static_id, support_metadata_id)
            VALUES ('Test Baikal Resource', %s, %s) RETURNING id
        """, (static_id, meta_id))
        resource_id = db_client.fetchone("SELECT lastval()")[0]
        db_client.execute("""
            INSERT INTO eco_assistant.modality (modality_type, value_table_name)
            VALUES ('Текст', 'text_value') ON CONFLICT (modality_type) DO NOTHING
        """)
        modality_row = db_client.fetchone("SELECT id FROM eco_assistant.modality WHERE modality_type = 'Текст'")
        modality_id = modality_row[0]
        db_client.execute("""
            INSERT INTO eco_assistant.text_value (structured_data) VALUES ('{}') RETURNING id
        """)
        text_value_id = db_client.fetchone("SELECT lastval()")[0]
        db_client.execute("""
            INSERT INTO eco_assistant.resource_value (resource_id, modality_id, value_id)
            VALUES (%s, %s, %s)
        """, (resource_id, modality_id, text_value_id))
        db_client.commit()
        criteria = ResourceCriteria(title='Baikal')
        results = search_repository.find_resources_by_criteria(criteria)
        assert len(results) >= 1
        assert any('Baikal' in (r.title or '') for r in results)