import pytest
from unittest.mock import Mock, MagicMock, patch
from sqlalchemy.orm import Session
from search_api.domain.entities import ObjectCriteria, ResourceCriteria, ObjectResult, ResourceResult
from search_api.adapters.sqlalchemy_repository import SQLAlchemySearchRepository


class TestSQLAlchemySearchRepository:
    """Unit tests for SQLAlchemySearchRepository"""
    
    @pytest.fixture
    def mock_session(self):
        session = Mock(spec=Session)
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        return session
    
    @pytest.fixture
    def mock_session_factory(self, mock_session):
        factory = Mock()
        factory.return_value = mock_session
        return factory
    
    @pytest.fixture
    def repository(self, mock_session_factory):
        return SQLAlchemySearchRepository(mock_session_factory)
    
    def test_find_objects_by_criteria_empty_criteria(self, repository, mock_session):
        """Test that empty criteria returns empty list without querying"""
        criteria = ObjectCriteria()  # All fields None
        
        result = repository.find_objects_by_criteria(criteria)
        
        assert result == []
        mock_session.query.assert_not_called()
    
    def test_find_objects_by_criteria_with_db_id(self, repository, mock_session):
        """Test filtering by db_id"""
        from search_api.infrastructure.orm.object_models import Object
        
        criteria = ObjectCriteria(db_id="test_db_123")
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        repository.find_objects_by_criteria(criteria)
        
        mock_session.query.assert_called_once()
        mock_query.filter.assert_called()
    
    def test_find_objects_by_criteria_with_object_type(self, repository, mock_session):
        """Test filtering by object_type"""
        from search_api.infrastructure.orm.object_models import ObjectType
        
        criteria = ObjectCriteria(object_type="Объект флоры и фауны")
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        repository.find_objects_by_criteria(criteria)
        
        mock_session.query.assert_called_once()
    
    def test_find_objects_by_criteria_with_name_synonyms(self, repository, mock_session):
        """Test filtering by name synonyms"""
        criteria = ObjectCriteria(
            name_synonyms={"ru": ["байкал", "озеро"]}
        )
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        repository.find_objects_by_criteria(criteria)
        
        mock_session.query.assert_called_once()
    
    def test_find_objects_by_criteria_with_properties(self, repository, mock_session):
        """Test filtering by properties"""
        criteria = ObjectCriteria(
            properties={"subtypes": "заповедник"}
        )
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        repository.find_objects_by_criteria(criteria)
        
        mock_session.query.assert_called_once()
    
    def test_find_objects_by_criteria_with_limit_offset(self, repository, mock_session):
        """Test limit and offset are applied correctly"""
        criteria = ObjectCriteria(db_id="test")
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        repository.find_objects_by_criteria(criteria, limit=10, offset=5)
        
        mock_query.limit.assert_called_with(10)
        mock_query.offset.assert_called_with(5)
    
    def test_find_resources_by_criteria_without_object_ids(self, repository, mock_session):
        """Test resource search without object_ids"""
        from search_api.infrastructure.orm.resource_models import Resource
        
        criteria = ResourceCriteria(title="test")
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.outerjoin.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        repository.find_resources_by_criteria(criteria)
        
        mock_session.query.assert_called_once()
    
    def test_find_resources_by_criteria_with_object_ids(self, repository, mock_session):
        """Test resource search with object_ids filter"""
        criteria = ResourceCriteria()
        object_ids = [1, 2, 3]
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.outerjoin.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        repository.find_resources_by_criteria(criteria, object_ids=object_ids)
        
        mock_session.query.assert_called_once()
    
    def test_find_resources_by_criteria_with_author(self, repository, mock_session):
        """Test filtering resources by author"""
        criteria = ResourceCriteria(author="Иванов")
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.outerjoin.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        repository.find_resources_by_criteria(criteria)
        
        mock_session.query.assert_called_once()
    
    def test_find_resources_by_criteria_with_source(self, repository, mock_session):
        """Test filtering resources by source"""
        criteria = ResourceCriteria(source="Научный журнал")
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.outerjoin.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        repository.find_resources_by_criteria(criteria)
        
        mock_session.query.assert_called_once()
    
    def test_find_resources_by_criteria_with_modality_type(self, repository, mock_session):
        """Test filtering resources by modality_type"""
        criteria = ResourceCriteria(modality_type="Текст")
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.outerjoin.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        repository.find_resources_by_criteria(criteria)
        
        mock_session.query.assert_called_once()
    
    def test_find_resources_by_criteria_with_features(self, repository, mock_session):
        """Test filtering resources by features"""
        criteria = ResourceCriteria(features={"rating": 5})
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.outerjoin.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        repository.find_resources_by_criteria(criteria)
        
        mock_session.query.assert_called_once()


class TestSQLAlchemySearchRepositoryIntegration:
    """Integration-like unit tests with mocked ORM objects"""
    
    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def repository(self, mock_session):
        def factory():
            return mock_session
        return SQLAlchemySearchRepository(factory)
    
    def test_find_objects_by_criteria_returns_objects(self, repository, mock_session):
        """Test that find_objects_by_criteria returns properly mapped ObjectResult objects"""
        from search_api.infrastructure.orm.object_models import Object, ObjectType
        
        # Create mock objects
        mock_obj = Mock(spec=Object)
        mock_obj.id = 1
        mock_obj.db_id = "test_db_001"
        mock_obj.object_properties = {"name": "Test Object"}
        mock_obj.object_type = Mock(spec=ObjectType)
        mock_obj.object_type.name = "Тестовый тип"
        mock_obj.synonyms = []
        
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = [mock_obj]
        
        criteria = ObjectCriteria(db_id="test_db_001")
        results = repository.find_objects_by_criteria(criteria)
        
        assert len(results) == 1
        assert isinstance(results[0], ObjectResult)
        assert results[0].id == 1
        assert results[0].db_id == "test_db_001"
        assert results[0].object_type == "Тестовый тип"
        assert results[0].properties == {"name": "Test Object"}
        assert results[0].synonyms == []