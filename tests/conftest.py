import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'knowledge_base_scripts' / 'Relational'))

import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch
from flask import Flask

from api import app as flask_app
from search_api.config import SearchConfig
from search_api.infrastructure.redis_cache import RedisCache
from search_api.adapters.database import PostgresSearchRepository
from search_api.adapters.search_repository import SearchRepository

from db_importer.config import DatabaseConfig
from db_importer.adapters import PostgresClient, PostgresSchemaRepository

os.environ.setdefault('DB_NAME', 'eco_test')

@pytest.fixture(scope='session')
def test_config() -> SearchConfig:
    return SearchConfig(
        db_name='eco_test',
        db_user='postgres',
        db_password='Fdf78yh0a4b!',
        db_host='localhost',
        db_port='5432',
        redis_host='localhost',
        redis_port=6379,
        redis_db=15,
        maps_dir='/tmp/test_maps',
        domain='http://localhost:5555'
    )

@pytest.fixture(scope='session')
def db_config() -> DatabaseConfig:
    return DatabaseConfig(
        dbname='eco_test',
        user='postgres',
        password='Fdf78yh0a4b!',
        host='localhost',
        port='5432'
    )

@pytest.fixture(scope='session', autouse=True)
def init_database(db_config: DatabaseConfig):
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    conn = psycopg2.connect(
        dbname='postgres',
        user=db_config.user,
        password=db_config.password,
        host=db_config.host,
        port=db_config.port
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_config.dbname}'")
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {db_config.dbname}")
    cur.close()
    conn.close()
    
    client = PostgresClient(db_config)
    client.connect()
    
    try:
        client.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        client.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        client.commit()
    except Exception as e:
        print(f"Warning: could not create extensions: {e}")
    
    schema_file = (project_root / 'knowledge_base_scripts' / 'Relational' / 'db_importer' / 'schema.sql')
    if schema_file.exists():
        schema_repo = PostgresSchemaRepository(client, schema_file)
        try:
            schema_repo.drop_all()
        except Exception:
            pass
        schema_repo.create_all()
    else:
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    
    client.disconnect()
    
    yield
    
    conn_drop = psycopg2.connect(
        dbname='postgres',
        user=db_config.user,
        password=db_config.password,
        host=db_config.host,
        port=db_config.port
    )
    conn_drop.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur_drop = conn_drop.cursor()
    cur_drop.execute(f"DROP DATABASE IF EXISTS {db_config.dbname} WITH (FORCE)")
    cur_drop.close()
    conn_drop.close()

@pytest.fixture
def db_client(db_config: DatabaseConfig, init_database):
    client = PostgresClient(db_config)
    client.connect()
    yield client
    client.disconnect()

@pytest.fixture
def mock_redis():
    mock = Mock(spec=RedisCache)
    mock.get.return_value = (False, None)
    mock.set.return_value = True
    mock.ping.return_value = True
    return mock

@pytest.fixture
def mock_llm():
    with patch('search_api.services.llm_answer_generator.get_llm') as mock:
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(
            content="Тестовый ответ LLM",
            response_metadata={'finish_reason': 'stop'}
        )
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def mock_geo_service():
    with patch('search_api.services.geo_map_service.GeoMapService') as mock:
        mock_instance = Mock()
        mock_instance.enrich_geo_content.return_value = Mock(
            geojson={'type': 'Point', 'coordinates': [104.3, 52.3]},
            geometry_type='Point',
            map_links=Mock(static='http://test/static.jpg', interactive='http://test/interactive.html')
        )
        mock_instance.generate_static_map.return_value = 'http://test/static.jpg'
        mock_instance.generate_interactive_map.return_value = 'http://test/interactive.html'
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def mock_repository():
    repo = Mock(spec=SearchRepository)
    repo.find_objects_by_criteria.return_value = []
    repo.find_resources_by_criteria.return_value = []
    return repo

@pytest.fixture
def app(mock_redis, mock_llm, mock_geo_service, mock_repository, test_config: SearchConfig):
    test_app = flask_app
    test_app.config['TESTING'] = True
    test_app.config['SEARCH_CONFIG'] = test_config
    test_app.config['SEARCH_REDIS'] = mock_redis
    test_app.config['SEARCH_REPOSITORY'] = mock_repository
    return test_app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def search_repository(db_client):
    return PostgresSearchRepository(
        SearchConfig(
            db_name='eco_test',
            db_user='postgres',
            db_password='Fdf78yh0a4b!',
            db_host='localhost',
            db_port='5432',
            redis_host='localhost',
            redis_port=6379,
            redis_db=15,
            maps_dir='/tmp/test_maps',
            domain='http://localhost:5555'
        )
    )
    
@pytest.fixture(scope='session')
def production_config() -> SearchConfig:
    return SearchConfig(
        db_name=os.getenv('DB_NAME', 'eco'),
        db_user=os.getenv('DB_USER', 'postgres'),
        db_password=os.getenv('DB_PASSWORD', 'Fdf78yh0a4b!'),
        db_host=os.getenv('DB_HOST', 'localhost'),
        db_port=os.getenv('DB_PORT', '5432'),
        redis_host=os.getenv('REDIS_HOST', 'localhost'),
        redis_port=int(os.getenv('REDIS_PORT', '6379')),
        redis_db=int(os.getenv('REDIS_DB', '1')),
        maps_dir=os.getenv('MAPS_DIR', '/app/maps'),
        domain=os.getenv('DOMAIN', 'http://localhost:5555')
    )

@pytest.fixture
def production_app(production_config):
    from api import app as flask_app
    flask_app.config['TESTING'] = True
    flask_app.config['SEARCH_CONFIG'] = production_config
    return flask_app

@pytest.fixture
def production_client(production_app):
    return production_app.test_client()