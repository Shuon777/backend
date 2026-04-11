import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'knowledge_base_scripts' / 'Relational'))

import pytest
from unittest.mock import Mock
from db_importer.use_cases import ImportObjectsUseCase
from db_importer.domain.entities import Object, ObjectType, DbId

class TestImportObjectsUseCase:
    def test_import_new_object(self):
        object_repo = Mock()
        object_type_repo = Mock()
        synonym_repo = Mock()
        
        use_case = ImportObjectsUseCase(object_repo, object_type_repo, synonym_repo)
        
        object_data = {
            'identificator': {'db_id': 'test_001'},
            'type': 'TestType',
            'properties': {'key': 'value'},
            'name_synonyms': {'ru_names': ['Тестовый объект']}
        }
        
        object_type = ObjectType(id=1, name='TestType')
        object_type_repo.get_or_create.return_value = object_type
        object_repo.find_by_db_id.return_value = None
        object_repo.save.return_value = Object(
            db_id=DbId('test_001'),
            object_type_id=1,
            object_properties={'key': 'value'},
            id=1
        )
        
        result = use_case.execute([object_data])
        
        assert result['created'] == 1
        assert result['updated'] == 0
        object_repo.save.assert_called_once()

    def test_import_existing_object(self):
        object_repo = Mock()
        object_type_repo = Mock()
        synonym_repo = Mock()
        
        use_case = ImportObjectsUseCase(object_repo, object_type_repo, synonym_repo)
        
        object_data = {
            'identificator': {'db_id': 'test_001'},
            'type': 'TestType',
            'properties': {'updated_key': 'updated_value'},
            'name_synonyms': {'ru_names': ['Тестовый объект']}
        }
        
        existing_object = Object(
            db_id=DbId('test_001'),
            object_type_id=1,
            object_properties={'old_key': 'old_value'},
            id=1
        )
        object_type = ObjectType(id=1, name='TestType')
        
        object_type_repo.get_or_create.return_value = object_type
        object_repo.find_by_db_id.return_value = existing_object
        object_repo.save.return_value = existing_object
        
        result = use_case.execute([object_data])
        
        assert result['created'] == 0
        assert result['updated'] == 1
        assert existing_object.object_properties == {'updated_key': 'updated_value'}

    def test_import_object_without_db_id(self):
        object_repo = Mock()
        object_type_repo = Mock()
        synonym_repo = Mock()
        
        use_case = ImportObjectsUseCase(object_repo, object_type_repo, synonym_repo)
        
        object_data = {
            'type': 'TestType',
            'properties': {}
        }
        
        result = use_case.execute([object_data])
        
        assert result['errors'] == 1
        object_repo.save.assert_not_called()