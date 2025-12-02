import os
import psycopg2
from psycopg2 import sql
import json
import sys
from pathlib import Path
from pgvector.psycopg2 import register_vector
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from embedding_config import embedding_config, get_model_dimension

class DatabaseRecreator:
    def __init__(self):
        self.db_config = {
            "dbname": os.getenv("DB_NAME", "eco"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "Fdf78yh0a4b!"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }
        
        self.species_synonyms_path = self._get_species_synonyms_path()
        self.species_synonyms = self._load_species_synonyms()
        
        self.connection = None
        self.cursor = None
        
        current_model = os.getenv("EMBEDDING_MODEL", embedding_config.current_model)
        embedding_dimension = os.getenv("EMBEDDING_DIMENSION")
        
        if embedding_dimension:
            self.embedding_dimension = int(embedding_dimension)
        else:
            self.embedding_dimension = get_model_dimension(current_model)
            
        self.embedding_model_path = embedding_config.get_model_path(current_model)
        
        print(f"📏 Размерность эмбеддингов: {self.embedding_dimension}")
        print(f"🎯 Активная модель: {current_model}")
        print(f"📁 Путь к модели: {self.embedding_model_path}")

    def _get_species_synonyms_path(self):
        """Определяем путь к файлу species_synonyms.json"""
        current_dir = Path(__file__).parent
        base_dir = current_dir.parent.parent
        json_files_dir = base_dir / "json_files"
        return json_files_dir / "species_synonyms.json"

    def _load_species_synonyms(self):
        """Загружает синонимы видов из JSON файла"""
        try:
            with open(self.species_synonyms_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Файл синонимов не найден: {self.species_synonyms_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON файла синонимов: {e}")
            return {}
        except Exception as e:
            print(f"Ошибка загрузки синонимов: {e}")
            return {}
    
    def connect(self):
        """Установка соединения с базой данных"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            print("Успешное подключение к базе данных")
        except Exception as e:
            print(f"Ошибка подключения к базе данных: {e}")
            raise

    def disconnect(self):
        """Закрытие соединения с базой данных"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("Соединение с базой данных закрыто")
        
    def execute_script(self, script):
        """Выполнение SQL-скрипта"""
        try:
            self.cursor.execute(script)
            self.connection.commit()
            print("Скрипт выполнен успешно")
        except Exception as e:
            self.connection.rollback()
            print(f"Ошибка выполнения скрипта: {e}")
            raise

    def drop_tables(self):
        """Удаление всех таблиц"""
        drop_script = """
        SET session_replication_role = replica;
        
        DROP TABLE IF EXISTS 
            public.ancient_human_made,
            public.audio_content,
            public.author,
            public.biological_entity,
            public.chart_content,
            public.conservation_action,
            public.document_content,
            public.ecological_reference,
            public.educational_content,
            public.entertainment_content,
            public.entity_author,
            public.entity_ecological,
            public.entity_geo,
            public.entity_identifier,
            public.entity_identifier_link,
            public.entity_park,
            public.entity_relation,
            public.entity_temporal,
            public.entity_territorial,
            public.entity_weather,
            public.external_link,
            public.geographical_entity,
            public.image_content,
            public.map_content,
            public.modern_human_made,
            public.organization,
            public.park_reference,
            public.reliability,
            public.research_project,
            public.route,
            public.stream_content,
            public.temporal_reference,
            public.territorial_reference,
            public.text_content,
            public.video_content,
            public.volunteer_initiative,
            public.weather_reference,
            public.error_log
        CASCADE;

        DROP EXTENSION IF EXISTS postgis CASCADE;

        SET session_replication_role = DEFAULT;
        """
        self.execute_script(drop_script)

    def create_tables(self):
        """Создание всех таблиц и индексов"""
        create_script = f"""
        CREATE EXTENSION IF NOT EXISTS postgis;
        CREATE EXTENSION IF NOT EXISTS vector;

        -- Универсальные идентификаторы сущностей
        CREATE TABLE entity_identifier (
            id SERIAL PRIMARY KEY,
            url VARCHAR(1000),
            db_path VARCHAR(500),
            file_path VARCHAR(500),
            name_ru VARCHAR(500),
            name_en VARCHAR(500),
            name_latin VARCHAR(500)
        );

        -- Новые типы контента
        CREATE TABLE document_content (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            file_format VARCHAR(50),
            feature_data JSONB
        );

        CREATE TABLE chart_content (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            chart_type VARCHAR(100),
            feature_data JSONB
        );

        CREATE TABLE stream_content (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            stream_url VARCHAR(1000) NOT NULL,
            schedule JSONB,
            feature_data JSONB
        );

        -- Таблица достоверности с указанием столбца
        CREATE TABLE reliability (
            id SERIAL PRIMARY KEY,
            entity_table VARCHAR(100) NOT NULL,
            entity_id INT NOT NULL,
            column_name VARCHAR(100),  -- Название столбца (если NULL - вся запись)
            reliability_value VARCHAR(50) NOT NULL,
            comment TEXT,
            CHECK (entity_table <> ''),
            UNIQUE (entity_table, entity_id, column_name)
        );

        -- Погодная привязка с временной меткой
        CREATE TABLE weather_reference (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            temperature_approx NUMERIC(5,2),
            temperature_feeling VARCHAR(50),
            windy BOOLEAN,
            rain BOOLEAN,
            humidity SMALLINT CHECK (humidity BETWEEN 0 AND 100),
            weather_conditions VARCHAR(255)
        );

        -- Парковая привязка
        CREATE TABLE park_reference (
            id SERIAL PRIMARY KEY,
            park_activity_type VARCHAR(50) NOT NULL,
            description TEXT
        );

        -- Экологическая привязка
        CREATE TABLE ecological_reference (
            id SERIAL PRIMARY KEY,
            ecosystem_features TEXT,
            restoration_methods TEXT,
            protection_regime TEXT,
            threats TEXT,
            anthropogenic_factors TEXT,
            ecological_disaster BOOLEAN
        );

        -- Территориальная привязка
        CREATE TABLE territorial_reference (
            id SERIAL PRIMARY KEY,
            territory_type VARCHAR(100) NOT NULL,
            description TEXT,
            natural_conditions TEXT
        );

        -- Основные сущности контента (без reliability)
        CREATE TABLE map_content (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            geometry GEOMETRY(Geometry, 4326) NOT NULL,
            feature_data JSONB
        );
        
        CREATE TABLE text_content (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500),
            content TEXT,  -- Убрано NOT NULL
            structured_data JSONB,
            description TEXT,
            feature_data JSONB,
            embedding vector({self.embedding_dimension})
        );
        
        CREATE TABLE image_content (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            feature_data JSONB
        );

        CREATE TABLE audio_content (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            duration INTERVAL,
            feature_data JSONB
        );

        CREATE TABLE video_content (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            duration INTERVAL,
            feature_data JSONB
        );

        CREATE TABLE geographical_entity (
            id SERIAL PRIMARY KEY,
            name_ru VARCHAR(500) NOT NULL,
            description TEXT,
            type VARCHAR(100),
            feature_data JSONB
        );

        CREATE TABLE biological_entity (
            id SERIAL PRIMARY KEY,
            common_name_ru VARCHAR(500) NOT NULL,
            scientific_name VARCHAR(500),
            description TEXT,
            status VARCHAR(100),
            type VARCHAR(100),
            feature_data JSONB
        );

        CREATE TABLE entertainment_content (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            feature_data JSONB
        );

        CREATE TABLE educational_content (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            feature_data JSONB
        );

        CREATE TABLE research_project (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            start_date DATE,
            end_date DATE,
            feature_data JSONB
        );

        CREATE TABLE organization (
            id SERIAL PRIMARY KEY,
            name_ru VARCHAR(500) NOT NULL,
            description TEXT,
            contacts JSONB,
            feature_data JSONB
        );

        CREATE TABLE route (
            id SERIAL PRIMARY KEY,
            name_ru VARCHAR(500) NOT NULL,
            description TEXT,
            length_km FLOAT,
            duration INTERVAL,
            difficulty VARCHAR(50),
            feature_data JSONB
        );

        CREATE TABLE modern_human_made (
            id SERIAL PRIMARY KEY,
            name_ru VARCHAR(500) NOT NULL,
            description TEXT,
            feature_data JSONB
        );

        CREATE TABLE ancient_human_made (
            id SERIAL PRIMARY KEY,
            name_ru VARCHAR(500) NOT NULL,
            description TEXT,
            historical_period VARCHAR(100),
            feature_data JSONB
        );

        CREATE TABLE volunteer_initiative (
            id SERIAL PRIMARY KEY,
            name_ru VARCHAR(500) NOT NULL,
            description TEXT,
            start_date DATE,
            end_date DATE,
            feature_data JSONB
        );

        CREATE TABLE conservation_action (
            id SERIAL PRIMARY KEY,
            name_ru VARCHAR(500) NOT NULL,
            description TEXT,
            start_date DATE,
            end_date DATE,
            feature_data JSONB
        );

        -- Вспомогательные сущности
        CREATE TABLE author (
            id SERIAL PRIMARY KEY,
            full_name VARCHAR(255) NOT NULL,
            organization VARCHAR(255),
            professional_scope TEXT
        );

        CREATE TABLE temporal_reference (
            id SERIAL PRIMARY KEY,
            resource_creation_date DATE,
            event_start_date DATE,
            event_end_date DATE,
            event_year INT,
            season VARCHAR(50),
            month SMALLINT CHECK (month BETWEEN 1 AND 12)
        );

        -- Универсальные связи
        CREATE TABLE entity_identifier_link (
            entity_id INT NOT NULL,
            entity_type VARCHAR(30) NOT NULL,
            identifier_id INT NOT NULL REFERENCES entity_identifier(id) ON DELETE CASCADE,
            PRIMARY KEY (entity_id, entity_type, identifier_id)
        );

        CREATE TABLE entity_author (
            entity_id INT NOT NULL,
            entity_type VARCHAR(30) NOT NULL,
            author_id INT NOT NULL REFERENCES author(id) ON DELETE CASCADE,
            PRIMARY KEY (entity_id, entity_type, author_id)
        );

        CREATE TABLE entity_temporal (
            entity_id INT NOT NULL,
            entity_type VARCHAR(30) NOT NULL,
            temporal_id INT NOT NULL REFERENCES temporal_reference(id) ON DELETE CASCADE,
            PRIMARY KEY (entity_id, entity_type, temporal_id)
        );

        CREATE TABLE entity_relation (
            source_id INT NOT NULL,
            source_type VARCHAR(30) NOT NULL,
            target_id INT NOT NULL,
            target_type VARCHAR(30) NOT NULL,
            relation_type VARCHAR(50) NOT NULL,
            PRIMARY KEY (source_id, source_type, target_id, target_type)
        );

        -- Географические привязки
        CREATE TABLE entity_geo (
            entity_id INT NOT NULL,
            entity_type VARCHAR(30) NOT NULL,
            geographical_entity_id INT NOT NULL REFERENCES geographical_entity(id) ON DELETE CASCADE,
            PRIMARY KEY (entity_id, entity_type, geographical_entity_id)
        );

        -- Территориальные привязки
        CREATE TABLE entity_territorial (
            entity_id INT NOT NULL,
            entity_type VARCHAR(30) NOT NULL,
            territorial_id INT NOT NULL REFERENCES territorial_reference(id) ON DELETE CASCADE,
            PRIMARY KEY (entity_id, entity_type, territorial_id)
        );

        -- Погодные привязки
        CREATE TABLE entity_weather (
            entity_id INT NOT NULL,
            entity_type VARCHAR(30) NOT NULL,
            weather_id INT NOT NULL REFERENCES weather_reference(id) ON DELETE CASCADE,
            PRIMARY KEY (entity_id, entity_type, weather_id)
        );

        -- Парковые привязки
        CREATE TABLE entity_park (
            entity_id INT NOT NULL,
            entity_type VARCHAR(30) NOT NULL,
            park_id INT NOT NULL REFERENCES park_reference(id) ON DELETE CASCADE,
            PRIMARY KEY (entity_id, entity_type, park_id)
        );

        -- Экологические привязки
        CREATE TABLE entity_ecological (
            entity_id INT NOT NULL,
            entity_type VARCHAR(30) NOT NULL,
            ecological_id INT NOT NULL REFERENCES ecological_reference(id) ON DELETE CASCADE,
            PRIMARY KEY (entity_id, entity_type, ecological_id)
        );

        CREATE TABLE external_link (
            id SERIAL PRIMARY KEY,
            url VARCHAR(1000) NOT NULL,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            link_type VARCHAR(50),
            platform VARCHAR(100)  -- Rutube, YouTube, VK и т.д.
        );
            -- Таблица для сбора ошибок
        CREATE TABLE error_log (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_query TEXT,
            error_message TEXT NOT NULL,
            context JSONB,
            additional_info JSONB
        );

        -- Индексы
        CREATE INDEX idx_error_log_created_at ON error_log (created_at);
        CREATE INDEX idx_error_log_context ON error_log USING GIN (context);
        CREATE INDEX idx_error_log_additional_info ON error_log USING GIN (additional_info);
        CREATE INDEX idx_external_link_type ON external_link (link_type);
        CREATE INDEX idx_external_link_platform ON external_link (platform);
        CREATE INDEX idx_map_geometry ON map_content USING GIST(geometry);
        CREATE INDEX idx_geo_name ON geographical_entity (name_ru);
        CREATE INDEX idx_bio_name ON biological_entity (common_name_ru);
        CREATE INDEX idx_reliability_entity ON reliability (entity_table, entity_id, column_name);
        CREATE INDEX idx_weather_time ON weather_reference (timestamp);
        CREATE INDEX idx_entity_geo_geographical_entity_id ON entity_geo(geographical_entity_id);
        CREATE INDEX idx_entity_geo_type_id ON entity_geo(entity_type, geographical_entity_id);
        CREATE INDEX idx_entity_geo_entity_id_type ON entity_geo(entity_id, entity_type);
        CREATE INDEX idx_biological_entity_scientific_name ON biological_entity(scientific_name);
        CREATE INDEX idx_geographical_entity_name ON geographical_entity(name_ru);
        CREATE INDEX idx_entity_geo_entity_type ON entity_geo(entity_type);
        CREATE INDEX idx_biological_entity_id ON biological_entity(id);
        CREATE INDEX idx_geographical_entity_id ON geographical_entity(id);
        CREATE INDEX idx_map_content_id ON map_content(id);

        CREATE INDEX idx_map_content_geometry_gist ON map_content USING GIST(geometry);
        CREATE INDEX idx_entity_geo_entity ON entity_geo(entity_type, entity_id);
        CREATE INDEX idx_text_content_structured_data ON text_content USING GIN (structured_data);
        
        CREATE INDEX idx_text_content_embedding ON text_content 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
        self.execute_script(create_script)

    def recreate_database(self):
        """Основной метод для пересоздания базы данных"""
        try:
            self.connect()
            self.drop_tables()
            self.create_tables()
            print("База данных успешно пересоздана")
        except Exception as e:
            print(f"Ошибка при пересоздании базы данных: {e}")
        finally:
            self.disconnect()

if __name__ == "__main__":
    db_recreator = DatabaseRecreator()
    db_recreator.recreate_database()