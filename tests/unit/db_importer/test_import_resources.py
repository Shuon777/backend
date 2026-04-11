import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'knowledge_base_scripts' / 'Relational'))

import pytest
from unittest.mock import Mock
from db_importer.use_cases import ImportResourcesUseCase
from db_importer.domain.entities import Resource, BibliographicData, CreationData, ResourceStatic, SupportMetadata

class TestImportResourcesUseCase:
    def test_import_new_resource(self):
        resource_repo = Mock()
        object_repo = Mock()
        resource_static_repo = Mock()
        metadata_repo = Mock()
        bibliographic_repo = Mock()
        creation_repo = Mock()
        modality_repo = Mock()
        geodata_provider = Mock()
        
        use_case = ImportResourcesUseCase(
            resource_repo, object_repo, resource_static_repo, metadata_repo,
            bibliographic_repo, creation_repo, modality_repo, geodata_provider
        )
        
        resource_data = {
            'title': 'Test Resource',
            'identificator': {'uri': 'http://test.com', 'id': 'test_001'},
            'bibliographic': {'author': 'Test Author', 'source': 'Test Source'},
            'creation': {'creation_type': 'manual'},
            'features': [{'name': 'feature1', 'value': 'value1'}],
            'modality': {'type': 'Текст', 'value': {'structured_data': {'key': 'value'}}}
        }
        
        bibliographic_repo.get_or_create_author.return_value = 1
        bibliographic_repo.get_or_create_source.return_value = 1
        bibliographic_repo.get_or_create_reliability_level.return_value = None
        bibliographic_repo.get_or_create.return_value = 1
        creation_repo.get_or_create.return_value = 1
        resource_static_repo.get_or_create.return_value = 1
        metadata_repo.get_or_create.return_value = 1
        modality_repo.get_or_create_modality.return_value = Mock(id=1)
        modality_repo.save_text_value.return_value = 1
        resource_repo.save_resource.return_value = 1
        
        result = use_case.execute([resource_data], incremental=False)
        
        assert result['success'] == 1
        resource_repo.save_resource.assert_called_once()

    def test_import_resource_with_incremental_skip(self):
        resource_repo = Mock()
        object_repo = Mock()
        resource_static_repo = Mock()
        metadata_repo = Mock()
        bibliographic_repo = Mock()
        creation_repo = Mock()
        modality_repo = Mock()
        geodata_provider = Mock()
        
        use_case = ImportResourcesUseCase(
            resource_repo, object_repo, resource_static_repo, metadata_repo,
            bibliographic_repo, creation_repo, modality_repo, geodata_provider
        )
        
        resource_data = {'title': 'Existing Resource'}
        resource_repo.resource_exists_by_hash.return_value = True
        
        result = use_case.execute([resource_data], incremental=True)
        
        assert result['skipped'] == 1
        resource_repo.save_resource.assert_not_called()