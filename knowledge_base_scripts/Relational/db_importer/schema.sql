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
    schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.object_type IS 'Справочник типов объектов';

-- ============================================================
-- 2. Основная таблица объектов
-- ============================================================
CREATE TABLE eco_assistant.object (
    id SERIAL PRIMARY KEY,
    db_id TEXT NOT NULL UNIQUE,
    object_type_id INTEGER NOT NULL REFERENCES eco_assistant.object_type(id),
    object_properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.object IS 'Объект';
CREATE INDEX idx_object_db_id ON eco_assistant.object(db_id);
CREATE INDEX idx_object_type ON eco_assistant.object(object_type_id);

-- ============================================================
-- 3. Синонимы названий объектов (связь многие-ко-многим)
-- ============================================================
CREATE TABLE eco_assistant.object_name_synonym (
    id SERIAL PRIMARY KEY,
    synonym TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'ru',
    created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.object_name_synonym IS 'Справочник синонимов названий объектов';

CREATE TABLE eco_assistant.object_name_synonym_link (
    object_id INTEGER NOT NULL REFERENCES eco_assistant.object(id) ON DELETE CASCADE,
    synonym_id INTEGER NOT NULL REFERENCES eco_assistant.object_name_synonym(id) ON DELETE CASCADE,
    PRIMARY KEY (object_id, synonym_id)
);
COMMENT ON TABLE eco_assistant.object_name_synonym_link IS 'Связь имен объектов с синонимами';

CREATE INDEX idx_synonym_trgm ON eco_assistant.object_name_synonym USING GIN (synonym gin_trgm_ops);
CREATE INDEX idx_synonym_mapping_object ON eco_assistant.object_name_synonym_link(object_id);
CREATE INDEX idx_synonym_mapping_synonym ON eco_assistant.object_name_synonym_link(synonym_id);

-- ============================================================
-- 4. Связи между объектами
-- ============================================================
CREATE TABLE eco_assistant.object_object_link (
    object_id INTEGER NOT NULL REFERENCES eco_assistant.object(id) ON DELETE CASCADE,
    related_object_id INTEGER NOT NULL REFERENCES eco_assistant.object(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (object_id, related_object_id, relation_type)
);
COMMENT ON TABLE eco_assistant.object_object_link IS 'Связи между объектами';

-- ============================================================
-- 5. Справочник модальностей
-- ============================================================
CREATE TABLE eco_assistant.modality (
    id SERIAL PRIMARY KEY,
    modality_type TEXT NOT NULL UNIQUE,
    value_table_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.modality IS 'Справочник модальностей ресурсов';

-- ============================================================
-- 6. Таблицы значений модальностей
-- ============================================================
CREATE TABLE eco_assistant.text_value (
    id SERIAL PRIMARY KEY,
    structured_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.text_value IS 'Значения текстовой модальности';

CREATE TABLE eco_assistant.image_value (
    id SERIAL PRIMARY KEY,
    url TEXT,
    file_path TEXT,
    format VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    CHECK (url IS NOT NULL OR file_path IS NOT NULL)
);
COMMENT ON TABLE eco_assistant.image_value IS 'Значения модальности "Изображение"';

CREATE TABLE eco_assistant.geodata_value (
    id SERIAL PRIMARY KEY,
    geometry GEOMETRY(Geometry, 4326) NOT NULL,
    geometry_type TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.geodata_value IS 'Значения модальности "Геоданные"';
CREATE INDEX idx_geodata_value_geom ON eco_assistant.geodata_value USING GIST(geometry);

-- ============================================================
-- 7. Связь ресурсов со значениями модальностей
-- ============================================================
CREATE TABLE eco_assistant.resource_value (
    id SERIAL PRIMARY KEY,
    resource_id INTEGER NOT NULL,
    modality_id INTEGER NOT NULL REFERENCES eco_assistant.modality(id),
    value_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(resource_id, modality_id)
);
COMMENT ON TABLE eco_assistant.resource_value IS 'Связь ресурса с модальностью и значением';

-- ============================================================
-- 8. Справочники для библиографических данных
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

CREATE TABLE eco_assistant.reliability_level (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.reliability_level IS 'Справочник уровней достоверности';

-- ============================================================
-- 9. Библиографические данные
-- ============================================================
CREATE TABLE eco_assistant.bibliographic (
    id SERIAL PRIMARY KEY,
    author_id INTEGER REFERENCES eco_assistant.author(id),
    date DATE,
    source_id INTEGER REFERENCES eco_assistant.source(id),
    reliability_level_id INTEGER REFERENCES eco_assistant.reliability_level(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.bibliographic IS 'Библиографические данные';

-- ============================================================
-- 10. Данные о создании
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
-- 11. Статические метаданные ресурса
-- ============================================================
CREATE TABLE eco_assistant.resource_static (
    id SERIAL PRIMARY KEY,
    static_id TEXT UNIQUE,
    bibliographic_id INTEGER NOT NULL REFERENCES eco_assistant.bibliographic(id),
    creation_id INTEGER NOT NULL REFERENCES eco_assistant.creation(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.resource_static IS 'Статические метаданные ресурса';

-- ============================================================
-- 12. Метаданные сопровождения
-- ============================================================
CREATE TABLE eco_assistant.support_metadata (
    id SERIAL PRIMARY KEY,
    parameters JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.support_metadata IS 'Метаданные сопровождения';

-- ============================================================
-- 13. Ресурс (центральная сущность)
-- ============================================================
CREATE TABLE eco_assistant.resource (
    id SERIAL PRIMARY KEY,
    title TEXT,
    uri TEXT,
    features JSONB,
    text_id TEXT UNIQUE,
    resource_static_id INTEGER NOT NULL REFERENCES eco_assistant.resource_static(id) ON DELETE CASCADE,
    support_metadata_id INTEGER NOT NULL REFERENCES eco_assistant.support_metadata(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_resource_text_id ON eco_assistant.resource(text_id);

-- ============================================================
-- 14. Связь ресурса с объектами
-- ============================================================
CREATE TABLE eco_assistant.resource_object (
    resource_id INTEGER NOT NULL REFERENCES eco_assistant.resource(id) ON DELETE CASCADE,
    object_id INTEGER NOT NULL REFERENCES eco_assistant.object(id) ON DELETE CASCADE,
    relation_type TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (resource_id, object_id)
);
COMMENT ON TABLE eco_assistant.resource_object IS 'Связь ресурса с объектами';

-- ============================================================
-- 15. Связь ресурса с ресурсами
-- ============================================================
CREATE TABLE eco_assistant.resource_resource_link (
    resource_id INTEGER NOT NULL REFERENCES eco_assistant.resource(id) ON DELETE CASCADE,
    related_resource_id INTEGER NOT NULL REFERENCES eco_assistant.resource(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (resource_id, related_resource_id, relation_type)
);
COMMENT ON TABLE eco_assistant.resource_resource_link IS 'Связи между ресурсами';

-- ============================================================
-- 16. Добавляем внешний ключ для resource_value после создания resource
-- ============================================================
ALTER TABLE eco_assistant.resource_value
    ADD CONSTRAINT fk_resource_value_resource
    FOREIGN KEY (resource_id) REFERENCES eco_assistant.resource(id) ON DELETE CASCADE;

-- ============================================================
-- 17. Свойства объектов (для фильтрации)
-- ============================================================
CREATE TABLE eco_assistant.object_property (
    id SERIAL PRIMARY KEY,
    object_type_id INTEGER NOT NULL REFERENCES eco_assistant.object_type(id) ON DELETE CASCADE,
    property_name TEXT NOT NULL,
    property_values TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(object_type_id, property_name)
);
COMMENT ON TABLE eco_assistant.object_property IS 'Свойства объектов для фильтрации';
CREATE INDEX idx_object_property_type ON eco_assistant.object_property(object_type_id);
CREATE INDEX idx_object_property_name ON eco_assistant.object_property(property_name);

-- ============================================================
-- 18. Признаки ресурсов (для фильтрации)
-- ============================================================
CREATE TABLE eco_assistant.resource_feature (
    id SERIAL PRIMARY KEY,
    modality_id INTEGER NOT NULL REFERENCES eco_assistant.modality(id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    feature_values TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(modality_id, feature_name)
);
COMMENT ON TABLE eco_assistant.resource_feature IS 'Признаки ресурсов для фильтрации';
CREATE INDEX idx_resource_feature_modality ON eco_assistant.resource_feature(modality_id);
CREATE INDEX idx_resource_feature_name ON eco_assistant.resource_feature(feature_name);

-- Добавление типов объектов по умолчанию
INSERT INTO eco_assistant.object_type (name, schema) VALUES 
('Объект флоры и фауны', '{}'),
('Географический объект', '{}'),
('Услуга', '{}'),
('Экспонат', '{}')
ON CONFLICT (name) DO NOTHING;