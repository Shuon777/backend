-- ============================================================
-- Установка необходимых расширений
-- ============================================================
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================
-- Создание схемы eco_assistant
-- ============================================================
CREATE SCHEMA IF NOT EXISTS eco_assistant;

-- ============================================================
-- 1. Справочник типов объектов
-- ============================================================
CREATE TABLE eco_assistant.object_type (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    schema JSONB NOT NULL DEFAULT '{}'::jsonb,        -- схема для заполнения свойств
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.object_type IS 'Справочник типов объектов';
COMMENT ON COLUMN eco_assistant.object_type.schema IS 'JSON-схема свойств объекта';

-- ============================================================
-- 2. Основная таблица объектов
-- ============================================================
CREATE TABLE eco_assistant.object (
    id SERIAL PRIMARY KEY,
    db_id TEXT NOT NULL UNIQUE,                       -- бывший canonical_id (хэш)
    object_type_id INTEGER NOT NULL REFERENCES eco_assistant.object_type(id),
    object_properties JSONB NOT NULL DEFAULT '{}'::jsonb,  -- произвольные свойства
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.object IS 'Объект (бывш. object_description)';
CREATE INDEX idx_object_db_id ON eco_assistant.object(db_id);
CREATE INDEX idx_object_type ON eco_assistant.object(object_type_id);

-- ============================================================
-- 3. Синонимы названий объектов (связь многие-ко-многим)
-- ============================================================
CREATE TABLE eco_assistant.object_name_synonym (
    id SERIAL PRIMARY KEY,
    synonym TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'ru',
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.object_name_synonym IS 'Справочник синонимов названий объектов';

CREATE TABLE eco_assistant.object_name_synonym_link (
    object_id INTEGER NOT NULL REFERENCES eco_assistant.object(id) ON DELETE CASCADE,
    synonym_id INTEGER NOT NULL REFERENCES eco_assistant.object_name_synonym(id) ON DELETE CASCADE,
    PRIMARY KEY (object_id, synonym_id)
);
COMMENT ON TABLE eco_assistant.object_name_synonym_link IS 'Связь имен объектов с синонимами';

-- Индексы для поиска
CREATE INDEX idx_synonym_trgm ON eco_assistant.object_name_synonym USING GIN (synonym gin_trgm_ops);
CREATE INDEX idx_synonym_mapping_object ON eco_assistant.object_name_synonym_link(object_id);
CREATE INDEX idx_synonym_mapping_synonym ON eco_assistant.object_name_synonym_link(synonym_id);

-- ============================================================
-- 4. Справочник модальностей
-- ============================================================
CREATE TABLE eco_assistant.modality (
    id SERIAL PRIMARY KEY,
    modality_type TEXT NOT NULL UNIQUE,
    value_table_name TEXT NOT NULL,                   -- имя таблицы со значениями (например 'text_value')
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.modality IS 'Справочник модальностей ресурсов';

-- ============================================================
-- 5. Таблицы значений модальностей
-- ============================================================
CREATE TABLE eco_assistant.text_value (
    id SERIAL PRIMARY KEY,
    content JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.text_value IS 'Значения текстовой модальности';

CREATE TABLE eco_assistant.image_value (
    id SERIAL PRIMARY KEY,
    url TEXT,
    file_path TEXT,
    quality VARCHAR(50),
    width INTEGER,
    height INTEGER,
    format VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    CHECK (url IS NOT NULL OR file_path IS NOT NULL)
);
COMMENT ON TABLE eco_assistant.image_value IS 'Значения модальности "Изображение"';

CREATE TABLE eco_assistant.geodata_value (
    id SERIAL PRIMARY KEY,
    geometry GEOMETRY(Geometry, 4326) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.geodata_value IS 'Значения модальности "Геоданные"';
CREATE INDEX idx_geodata_value_geom ON eco_assistant.geodata_value USING GIST(geometry);

-- ============================================================
-- 6. Связь ресурсов со значениями модальностей
-- ============================================================
CREATE TABLE eco_assistant.resource_value (
    id SERIAL PRIMARY KEY,
    resource_id INTEGER NOT NULL,                     -- будет добавлено после создания resource
    modality_id INTEGER NOT NULL REFERENCES eco_assistant.modality(id),
    value_id INTEGER,  -- id в таблице значений (text_value, image_value, geodata_value)
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(resource_id, modality_id)                  -- один ресурс - одна модальность
);
COMMENT ON TABLE eco_assistant.resource_value IS 'Связь ресурса с модальностью и значением';

-- ============================================================
-- 7. Справочники для библиографических данных
-- ============================================================
CREATE TABLE eco_assistant.author (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.author IS 'Справочник авторов';

CREATE TABLE eco_assistant.source (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.source IS 'Справочник источников';

CREATE TABLE eco_assistant.usage_right (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.usage_right IS 'Справочник прав использования';

CREATE TABLE eco_assistant.reliability_level (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.reliability_level IS 'Справочник уровней достоверности';

-- ============================================================
-- 8. Библиографические данные
-- ============================================================
CREATE TABLE eco_assistant.bibliographic (
    id SERIAL PRIMARY KEY,
    author_id INTEGER REFERENCES eco_assistant.author(id),
    date DATE,
    source_id INTEGER REFERENCES eco_assistant.source(id),
    usage_right_id INTEGER REFERENCES eco_assistant.usage_right(id),
    reliability_level_id INTEGER REFERENCES eco_assistant.reliability_level(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.bibliographic IS 'Библиографические данные';

-- ============================================================
-- 9. Данные о создании (бывш. generation)
-- ============================================================
CREATE TABLE eco_assistant.creation (
    id SERIAL PRIMARY KEY,
    creation_type TEXT,
    creation_tool TEXT,
    creation_params JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.creation IS 'Данные о создании (источник)';

-- ============================================================
-- 10. Статические метаданные ресурса
-- ============================================================
CREATE TABLE eco_assistant.resource_static (
    id SERIAL PRIMARY KEY,
    bibliographic_id INTEGER NOT NULL REFERENCES eco_assistant.bibliographic(id),
    creation_id INTEGER NOT NULL REFERENCES eco_assistant.creation(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.resource_static IS 'Статические метаданные ресурса';

-- ============================================================
-- 11. Метаданные сопровождения
-- ============================================================
CREATE TABLE eco_assistant.support_metadata (
    id SERIAL PRIMARY KEY,
    parameters JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.support_metadata IS 'Метаданные сопровождения';

-- ============================================================
-- 12. Ресурс (центральная сущность)
-- ============================================================
CREATE TABLE eco_assistant.resource (
    id SERIAL PRIMARY KEY,
    resource_static_id INTEGER NOT NULL REFERENCES eco_assistant.resource_static(id) ON DELETE CASCADE,
    support_metadata_id INTEGER NOT NULL REFERENCES eco_assistant.support_metadata(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.resource IS 'Ресурс';

-- ============================================================
-- 13. Связь ресурса с объектами (многие ко многим)
-- ============================================================
CREATE TABLE eco_assistant.resource_object (
    resource_id INTEGER NOT NULL REFERENCES eco_assistant.resource(id) ON DELETE CASCADE,
    object_id INTEGER NOT NULL REFERENCES eco_assistant.object(id) ON DELETE CASCADE,
    PRIMARY KEY (resource_id, object_id)
);
COMMENT ON TABLE eco_assistant.resource_object IS 'Связь ресурса с объектами';

-- ============================================================
-- 14. Признаки ресурса (онтология) в формате JSON
-- ============================================================
CREATE TABLE eco_assistant.feature_json (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    data JSONB NOT NULL,                              -- вместо description
    created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.feature_json IS 'Словарь признаков ресурса в формате JSON';

-- ============================================================
-- 15. Связь ресурса с признаками
-- ============================================================
CREATE TABLE eco_assistant.resource_feature (
    resource_id INTEGER NOT NULL REFERENCES eco_assistant.resource(id) ON DELETE CASCADE,
    feature_id INTEGER NOT NULL REFERENCES eco_assistant.feature_json(id) ON DELETE CASCADE,
    PRIMARY KEY (resource_id, feature_id)
);
COMMENT ON TABLE eco_assistant.resource_feature IS 'Связь ресурса с признаками';

-- ============================================================
-- 16. Обратная ссылка: ресурс -> значение модальности (добавляем foreign key после создания resource)
-- ============================================================
ALTER TABLE eco_assistant.resource_value
    ADD CONSTRAINT fk_resource_value_resource
    FOREIGN KEY (resource_id) REFERENCES eco_assistant.resource(id) ON DELETE CASCADE;