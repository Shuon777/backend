-- ============================================================
-- Установка необходимых расширений
-- ============================================================
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- для триграммного поиска
CREATE EXTENSION IF NOT EXISTS postgis;     -- для геоданных (если еще не установлен)

-- ============================================================
-- Создание схемы eco_assistant для информационной модели
-- ============================================================
CREATE SCHEMA IF NOT EXISTS eco_assistant;

-- ============================================================
-- Таблицы в схеме eco_assistant
-- ============================================================

-- -----------------------------------------------
-- 1. Базовый объект (ИдентификаторБД)
-- -----------------------------------------------
CREATE TABLE eco_assistant.object (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.object IS 'Базовый объект (ИдентификаторБД)';

-- -----------------------------------------------
-- 2. Описание связанного объекта
-- -----------------------------------------------
CREATE TABLE eco_assistant.object_description (
    id SERIAL PRIMARY KEY,
    object_id INTEGER NOT NULL REFERENCES eco_assistant.object(id) ON DELETE CASCADE,
    classification_identifier TEXT,
    object_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.object_description IS 'Описание объекта: ИдентификаторБД (object_id), ИдентификаторСТ, Тип объекта';
CREATE INDEX idx_obj_desc_object ON eco_assistant.object_description(object_id);
CREATE INDEX idx_obj_desc_type ON eco_assistant.object_description(object_type);

-- -----------------------------------------------
-- 3. Синонимы названия объекта
-- -----------------------------------------------
CREATE TABLE eco_assistant.object_synonym (
    id SERIAL PRIMARY KEY,
    object_description_id INTEGER NOT NULL REFERENCES eco_assistant.object_description(id) ON DELETE CASCADE,
    synonym TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'ru',
    UNIQUE(object_description_id, synonym, language)
);
COMMENT ON TABLE eco_assistant.object_synonym IS 'Синонимы названия объекта (множество)';
CREATE INDEX idx_obj_syn ON eco_assistant.object_synonym(synonym);

-- -----------------------------------------------
-- 4. Канонические значения свойств объектов
-- -----------------------------------------------
CREATE TABLE eco_assistant.property_value (
    id SERIAL PRIMARY KEY,
    value TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.property_value IS 'Канонические текстовые значения свойств';

-- -----------------------------------------------
-- 5. Синонимы значений свойств
-- -----------------------------------------------
CREATE TABLE eco_assistant.property_value_synonym (
    id SERIAL PRIMARY KEY,
    property_value_id INTEGER NOT NULL REFERENCES eco_assistant.property_value(id) ON DELETE CASCADE,
    synonym TEXT NOT NULL,
    UNIQUE(property_value_id, synonym)
);
COMMENT ON TABLE eco_assistant.property_value_synonym IS 'Синонимы для значений свойств';
CREATE INDEX idx_prop_val_syn ON eco_assistant.property_value_synonym(synonym);

-- -----------------------------------------------
-- 6. Свойства объекта по типам
-- -----------------------------------------------
CREATE TABLE eco_assistant.object_property (
    id SERIAL PRIMARY KEY,
    object_description_id INTEGER NOT NULL REFERENCES eco_assistant.object_description(id) ON DELETE CASCADE,
    property_name TEXT NOT NULL,
    object_type TEXT NOT NULL,
    property_value_id INTEGER NOT NULL REFERENCES eco_assistant.property_value(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(object_description_id, property_name, object_type, property_value_id)
);
COMMENT ON TABLE eco_assistant.object_property IS 'Свойства объекта (гибкие) с привязкой к типу объекта';
CREATE INDEX idx_obj_prop_desc ON eco_assistant.object_property(object_description_id);
CREATE INDEX idx_obj_prop_name_type ON eco_assistant.object_property(property_name, object_type);
CREATE INDEX idx_obj_prop_value ON eco_assistant.object_property(property_value_id);

-- -----------------------------------------------
-- 7. Модальность ресурса (базовая)
-- -----------------------------------------------
CREATE TABLE eco_assistant.modality (
    id SERIAL PRIMARY KEY,
    modality_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.modality IS 'Базовая модальность ресурса';

-- -----------------------------------------------
-- 8. Данные текстовой модальности
-- -----------------------------------------------
CREATE TABLE eco_assistant.modality_text (
    id SERIAL PRIMARY KEY,
    modality_id INTEGER NOT NULL REFERENCES eco_assistant.modality(id) ON DELETE CASCADE,
    content JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.modality_text IS 'Данные для текстовой модальности';
CREATE INDEX idx_modality_text ON eco_assistant.modality_text(modality_id);

-- -----------------------------------------------
-- 9. Данные модальности "Изображение"
-- -----------------------------------------------
CREATE TABLE eco_assistant.modality_image (
    id SERIAL PRIMARY KEY,
    modality_id INTEGER NOT NULL REFERENCES eco_assistant.modality(id) ON DELETE CASCADE,
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
COMMENT ON TABLE eco_assistant.modality_image IS 'Данные для модальности "Изображение"';
CREATE INDEX idx_modality_image ON eco_assistant.modality_image(modality_id);

-- -----------------------------------------------
-- 10. Данные модальности "Геоданные"
-- -----------------------------------------------
CREATE TABLE eco_assistant.modality_geodata (
    id SERIAL PRIMARY KEY,
    modality_id INTEGER NOT NULL REFERENCES eco_assistant.modality(id) ON DELETE CASCADE,
    geometry GEOMETRY(Geometry, 4326) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.modality_geodata IS 'Данные для модальности "Геоданные"';
CREATE INDEX idx_modality_geodata_geom ON eco_assistant.modality_geodata USING GIST(geometry);
CREATE INDEX idx_modality_geodata ON eco_assistant.modality_geodata(modality_id);

-- -----------------------------------------------
-- 11. Признак ресурса (словарь онтологии)
-- -----------------------------------------------
CREATE TABLE eco_assistant.feature (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.feature IS 'Словарь признаков ресурса (онтология)';

-- -----------------------------------------------
-- 12. Библиографические данные
-- -----------------------------------------------
CREATE TABLE eco_assistant.bibliographic (
    id SERIAL PRIMARY KEY,
    author TEXT,
    date DATE,
    source TEXT,
    rights TEXT,
    reliability VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.bibliographic IS 'Библиографические данные';

-- -----------------------------------------------
-- 13. Данные о генерации
-- -----------------------------------------------
CREATE TABLE eco_assistant.generation (
    id SERIAL PRIMARY KEY,
    generation_type TEXT,
    generation_tool TEXT,
    generation_params JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.generation IS 'Данные о генерации';

-- -----------------------------------------------
-- 14. Метаданные сопровождения
-- -----------------------------------------------
CREATE TABLE eco_assistant.support_metadata (
    id SERIAL PRIMARY KEY,
    parameters JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.support_metadata IS 'Метаданные сопровождения (параметры обработки)';

-- -----------------------------------------------
-- 15. Ресурс (центральная сущность)
-- -----------------------------------------------
CREATE TABLE eco_assistant.resource (
    id SERIAL PRIMARY KEY,
    modality_id INTEGER NOT NULL REFERENCES eco_assistant.modality(id) ON DELETE CASCADE,
    bibliographic_id INTEGER NOT NULL REFERENCES eco_assistant.bibliographic(id) ON DELETE CASCADE,
    generation_id INTEGER NOT NULL REFERENCES eco_assistant.generation(id) ON DELETE CASCADE,
    support_metadata_id INTEGER NOT NULL REFERENCES eco_assistant.support_metadata(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE eco_assistant.resource IS 'Ресурс: объединяет модальность, статические метаданные и метаданные сопровождения';

-- -----------------------------------------------
-- 16. Связь ресурса с описаниями объектов (многие ко многим)
-- -----------------------------------------------
CREATE TABLE eco_assistant.resource_object (
    resource_id INTEGER NOT NULL REFERENCES eco_assistant.resource(id) ON DELETE CASCADE,
    object_description_id INTEGER NOT NULL REFERENCES eco_assistant.object_description(id) ON DELETE CASCADE,
    PRIMARY KEY (resource_id, object_description_id)
);
COMMENT ON TABLE eco_assistant.resource_object IS 'Связь ресурса с множеством описаний объектов';

-- -----------------------------------------------
-- 17. Связь ресурса с признаками (многие ко многим)
-- -----------------------------------------------
CREATE TABLE eco_assistant.resource_feature (
    resource_id INTEGER NOT NULL REFERENCES eco_assistant.resource(id) ON DELETE CASCADE,
    feature_id INTEGER NOT NULL REFERENCES eco_assistant.feature(id) ON DELETE CASCADE,
    PRIMARY KEY (resource_id, feature_id)
);
COMMENT ON TABLE eco_assistant.resource_feature IS 'Связь ресурса с признаками (необязательно)';

-- ============================================================
-- Индексы для производительности
-- ============================================================
-- Для поиска по названиям объектов с использованием триграмм
CREATE INDEX idx_object_name_trgm ON eco_assistant.object USING GIN (name gin_trgm_ops);
CREATE INDEX idx_object_synonym_trgm ON eco_assistant.object_synonym USING GIN (synonym gin_trgm_ops);

-- Основные индексы для связей и поиска
CREATE INDEX idx_object_description_class ON eco_assistant.object_description(classification_identifier);
CREATE INDEX idx_object_property_value ON eco_assistant.object_property(property_value_id);
CREATE INDEX idx_modality_type ON eco_assistant.modality(modality_type);
CREATE INDEX idx_resource_modality ON eco_assistant.resource(modality_id);
CREATE INDEX idx_resource_bibliographic ON eco_assistant.resource(bibliographic_id);
CREATE INDEX idx_resource_generation ON eco_assistant.resource(generation_id);
CREATE INDEX idx_resource_support ON eco_assistant.resource(support_metadata_id);

-- ============================================================
-- Комментарии к колонкам для документации
-- ============================================================
COMMENT ON COLUMN eco_assistant.object_description.object_id IS 'ИдентификаторБД – ссылка на базовый объект';
COMMENT ON COLUMN eco_assistant.object_description.classification_identifier IS 'ИдентификаторСТ – идентификатор унифицированной классификации';
COMMENT ON COLUMN eco_assistant.object_description.object_type IS 'Тип объекта: ОФФ, геообъект, достопримечательность, услуги';
COMMENT ON COLUMN eco_assistant.modality.modality_type IS 'Тип модальности: Текст, Изображение, Геоданные, Аудио и т.д.';
COMMENT ON COLUMN eco_assistant.modality_text.content IS 'Структурированный текст (JSON)';
COMMENT ON COLUMN eco_assistant.modality_image.url IS 'Ссылка на изображение';
COMMENT ON COLUMN eco_assistant.modality_image.file_path IS 'Путь к файлу изображения';
COMMENT ON COLUMN eco_assistant.modality_image.quality IS 'Качество изображения';
COMMENT ON COLUMN eco_assistant.modality_geodata.geometry IS 'Геометрия (точки, полигоны, линии)';
COMMENT ON COLUMN eco_assistant.bibliographic.author IS 'Автор';
COMMENT ON COLUMN eco_assistant.bibliographic.date IS 'Дата';
COMMENT ON COLUMN eco_assistant.bibliographic.source IS 'Источник';
COMMENT ON COLUMN eco_assistant.bibliographic.rights IS 'Права использования';
COMMENT ON COLUMN eco_assistant.bibliographic.reliability IS 'Уровень достоверности';
COMMENT ON COLUMN eco_assistant.generation.generation_type IS 'Тип генерации';
COMMENT ON COLUMN eco_assistant.generation.generation_tool IS 'Средство генерации';
COMMENT ON COLUMN eco_assistant.generation.generation_params IS 'Параметры генерации';
COMMENT ON COLUMN eco_assistant.support_metadata.parameters IS 'Параметры для правильной интерпретации ресурса (JSON)';