import os
import json
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
from dotenv import load_dotenv
import re
from datetime import datetime
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from embedding_config import embedding_config, get_model_dimension

class NewResourceImporter:
    def __init__(self):
        self.db_config = {
            "dbname": os.getenv("DB_NAME", "eco"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "Fdf78yh0a4b!"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }
        self.missing_geometry_objects = set()
        current_model = os.getenv("EMBEDDING_MODEL", embedding_config.current_model)
        embedding_dimension = os.getenv("EMBEDDING_DIMENSION")
        
        if embedding_dimension:
            self.embedding_dimension = int(embedding_dimension)
        else:
            self.embedding_dimension = get_model_dimension(current_model)
            
        current_dir = Path(__file__).parent
        base_dir = current_dir.parent.parent
        embedding_models_dir = base_dir / "embedding_models" / "BERTA"
        
        self.embedding_model_path = str(embedding_models_dir)
        
        print(f"📏 Размерность эмбеддингов: {self.embedding_dimension}")
        print(f"🎯 Активная модель: {current_model}")
        print(f"📁 Путь к модели: {self.embedding_model_path}")
        
        
        self.conn = None
        self.cur = None
        self.entity_cache = {}
        self.author_cache = {}
        self.bio_entity_cache = {}
        self.geodb_data = self.load_geodb()
        self.species_synonyms_path = self._get_species_synonyms_path()
        self.species_synonyms = self.load_species_synonyms() or {}
        self.embedding_model = self.load_embedding_model()
    def safe_convert_in_stoplist(self, value):
        """Безопасно преобразует in_stoplist в число"""
        if value is None:
            return None
        if isinstance(value, bool):
            return 1 if value else 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 1  # значение по умолчанию
    def load_embedding_model(self):
        """Загрузка модели для генерации эмбеддингов"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            
            # Проверяем, что модель работает
            test_embedding = embeddings.embed_query("test")
            if test_embedding is None or len(test_embedding) == 0:
                raise Exception("Model loaded but returned empty embedding")
                
            print(f"✅ Модель эмбеддингов успешно загружена, размерность: {len(test_embedding)}")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            return None

    def generate_embedding(self, text):
        """Генерация эмбеддинга для текста"""
        if not text:
            print("⚠️  Пустой текст для эмбеддинга")
            return None
        
        if not self.embedding_model:
            print("❌ Модель эмбеддингов не загружена")
            return None
        
        try:
            combined_text = text
            embedding = self.embedding_model.embed_query(combined_text)
            
            if embedding is None:
                print("❌ Модель вернула None")
                return None
                
            if len(embedding) != self.embedding_dimension:
                print(f"⚠️  Предупреждение: Размерность эмбеддинга ({len(embedding)}) не совпадает с ожидаемой ({self.embedding_dimension})")
            
            print(f"✅ Сгенерирован эмбеддинг размерности {len(embedding)}")
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
        
    def load_geodb(self):
        try:
            with open("/var/www/salut_bot/json_files/geodb.json", 'r') as f:
                return json.load(f)
        except:
            return {}   
    
    def connect(self):
        self.conn = psycopg2.connect(**self.db_config)
        self.cur = self.conn.cursor()

    def disconnect(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
            
    def get_geo_data(self, geo_name):
        """Получаем полные геоданные для объекта из geodb.json с учетом частичных совпадений"""
        if not hasattr(self, 'geodb_data'):
            try:
                with open("/var/www/salut_bot/json_files/geodb.json", 'r') as f:
                    self.geodb_data = json.load(f)
            except Exception as e:
                print(f"Error loading geodb.json: {e}")
                return None
        
        # Поиск по точному соответствию
        if geo_name in self.geodb_data:
            return self.geodb_data[geo_name]
        
        # Поиск без учета регистра
        for name, data in self.geodb_data.items():
            if name.lower() == geo_name.lower():
                return data
        
        # Поиск частичных совпадений (без уточнения района)
        # Например: "Ольхонский район, мыс Бурхан" -> ищем "мыс Бурхан"
        geo_name_lower = geo_name.lower()
        
        # Пробуем найти часть после запятой
        if ',' in geo_name:
            parts = [part.strip() for part in geo_name.split(',')]
            # Ищем самые конкретные части (последние)
            for part in reversed(parts):
                if part and part in self.geodb_data:
                    return self.geodb_data[part]
                # Поиск без учета регистра
                for name, data in self.geodb_data.items():
                    if name.lower() == part.lower():
                        return data
        
        # Поиск по частичному вхождению (если есть общие слова)
        geo_words = set(geo_name_lower.split())
        best_match = None
        best_score = 0
        
        for name, data in self.geodb_data.items():
            name_lower = name.lower()
            name_words = set(name_lower.split())
            
            # Вычисляем степень совпадения
            common_words = geo_words.intersection(name_words)
            score = len(common_words)
            
            # Предпочтение более длинным совпадениям
            if score > best_score:
                best_score = score
                best_match = data
        
        if best_score >= 2:  # Минимум 2 общих слова
            return best_match
        
        # ВАЖНО: НЕ добавляем в missing_geometry_objects здесь!
        return None
    
    def process_geo_mention(self, source_id, source_type, geo_name, name_info):
        if not geo_name:
            return None
            
        try:
            normalized_name = self.normalize_geo_name(geo_name)
            
            # Ищем по нормализованному имени
            self.cur.execute(
                "SELECT id FROM geographical_entity "
                "WHERE lower(name_ru) = %s",
                (normalized_name,)
            )
            existing_geo = self.cur.fetchone()
            
            geo_id = None
            if existing_geo:
                geo_id = existing_geo[0]
            else:
                # Пробуем найти упрощенное название (без района)
                simplified_name = self.simplify_geo_name(geo_name)
                if simplified_name != normalized_name:
                    self.cur.execute(
                        "SELECT id FROM geographical_entity "
                        "WHERE lower(name_ru) = %s",
                        (simplified_name.lower(),)
                    )
                    existing_simplified = self.cur.fetchone()
                    if existing_simplified:
                        geo_id = existing_simplified[0]
            
            if not geo_id:
                # Создаем новую географическую сущность
                self.cur.execute(
                    "INSERT INTO geographical_entity (name_ru, feature_data) "
                    "VALUES (%s, %s) RETURNING id",
                    (geo_name, Json({
                        'source': 'text_mention',
                        'normalized_name': normalized_name,
                        'original_name': geo_name,
                        'simplified_name': self.simplify_geo_name(geo_name)
                    }))
                )
                geo_id = self.cur.fetchone()[0]

                self.add_reliability('geographical_entity', geo_id, name_info.get('source'))

            if source_id and source_type:
                self.cur.execute(
                    "INSERT INTO entity_geo (entity_id, entity_type, geographical_entity_id) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (entity_id, entity_type, geographical_entity_id) DO NOTHING",
                    (source_id, source_type, geo_id)
                )

            return geo_id

        except Exception as e:
            print(f"Error processing geo mention '{geo_name}': {e}")
            return None

    def simplify_geo_name(self, geo_name):
        """Упрощает название географического объекта, убирая указания районов"""
        if not geo_name:
            return geo_name
        
        # Убираем указания районов (все что до первой запятой)
        if ',' in geo_name:
            parts = [part.strip() for part in geo_name.split(',')]
            # Берем самую конкретную часть (обычно последнюю)
            return parts[-1]
        
        return geo_name.strip()

    def save_missing_geometry_objects(self, output_file="missing_geometry_objects.json"):
        """Сохраняет названия объектов без геометрии в JSON файл"""
        if self.missing_geometry_objects:
            missing_list = list(self.missing_geometry_objects)
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(missing_list, f, ensure_ascii=False, indent=2)
                print(f"Сохранено {len(missing_list)} объектов без геометрии в {output_file}")
            except Exception as e:
                print(f"Ошибка сохранения файла отсутствующих геометрий: {e}")
        else:
            print("Все гео-объекты имеют геометрию")
        
    def clean_coordinate(self, coord):
        """Очистка и валидация координат"""
        if coord is None:
            return None
        
        if isinstance(coord, (int, float)):
            return float(coord)
        
        if isinstance(coord, str):
            try:
                return float(coord)
            except ValueError:
                cleaned = coord.strip()
                try:
                    return float(cleaned)
                except ValueError:
                    print(f"Warning: Cannot convert coordinate '{coord}' to float")
                    return None
        
        try:
            return float(str(coord))
        except (ValueError, TypeError):
            print(f"Warning: Invalid coordinate type: {type(coord)}, value: {coord}")
            return None
        
    def _get_species_synonyms_path(self):
        """Определяем путь к файлу species_synonyms.json"""
        current_dir = Path(__file__).parent
        base_dir = current_dir.parent.parent
        json_files_dir = base_dir / "json_files"
        return json_files_dir / "species_synonyms.json"
    
    def load_species_synonyms(self):
        """Загрузка синонимов видов из JSON-файла"""
        try:
            with open(self.species_synonyms_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Файл синонимов {self.species_synonyms_path} не найден. Будет использован пустой словарь.")
            return {}
        except Exception as e:
            print(f"Ошибка загрузки файла синонимов: {e}")
            return {}
    
    def normalize_species_name(self, name):
        """Нормализация названия вида с учетом синонимов"""
        if not name:
            return name
            
        name_lower = name.strip().lower()
        for main_name, synonyms in self.species_synonyms.items():
            if name_lower in [s.lower() for s in synonyms] or name_lower == main_name.lower():
                return main_name
        return name
    
    def parse_date(self, date_str):
        """Преобразование даты в формат PostgreSQL"""
        if not date_str:
            return None
            
        try:
            # Удаляем специальные символы (· и т.д.)
            date_str = re.sub(r'[·•]', ' ', date_str).strip()
            
            # Пробуем разные форматы дат
            formats = [
                '%d.%m.%Y %H:%M',  # 24.05.2022 18:53
                '%d.%m.%Y',         # 24.05.2022
                '%d.%m.%y %H:%M',   # 24.05.22 18:53
                '%d.%m.%y',          # 24.05.22
                '%Y-%m-%d %H:%M:%S', # Стандартный SQL
                '%Y-%m-%d',          # Стандартный SQL (без времени)
                '%d/%m/%Y %H:%M',    # Альтернативный формат
                '%d/%m/%Y',          # Альтернативный формат
                '%d %m %Y %H:%M',    # Еще один вариант
                '%d %m %Y'           # Еще один вариант
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                    
            return None
        except Exception as e:
            print(f"Date parsing error for '{date_str}': {e}")
            return None

    def get_or_create_author(self, full_name, organization=None):
        """Получаем или создаем автора с проверкой существования"""
        if not full_name:
            return None
            
        cache_key = f"{full_name}_{organization}"
        if cache_key in self.author_cache:
            # Проверяем, что автор действительно существует
            author_id = self.author_cache[cache_key]
            self.cur.execute("SELECT 1 FROM author WHERE id = %s", (author_id,))
            if self.cur.fetchone():
                return author_id
            else:
                del self.author_cache[cache_key]
        
        try:
            # Ищем автора в базе
            self.cur.execute(
                "SELECT id FROM author WHERE full_name = %s AND organization = %s",
                (full_name, organization)
            )
            author = self.cur.fetchone()
            
            if author:
                self.author_cache[cache_key] = author[0]
                return author[0]
                
            # Создаем нового автора
            self.cur.execute(
                "INSERT INTO author (full_name, organization) VALUES (%s, %s) RETURNING id",
                (full_name, organization)
            )
            author_id = self.cur.fetchone()[0]
            self.conn.commit()  # Фиксируем создание автора сразу
            self.author_cache[cache_key] = author_id
            return author_id
            
        except Exception as e:
            print(f"Error processing author {full_name}: {e}")
            self.conn.rollback()
            return None

    def get_reliability_value(self, source):
        """Определяем уровень достоверности на основе источника"""
        if not source:
            return "общедоступная"
        
        source_lower = source.lower()
        if "национальный парк" in source_lower or "заповедник" in source_lower:
            return "профильная организация"
        elif "ai generation" in source_lower or "википедия" in source_lower:
            return "общедоступная"
        return "профильная организация"

    def add_reliability(self, table_name, entity_id, source, column_name=None):
        """Добавляем запись о достоверности"""
        reliability_value = self.get_reliability_value(source)
        try:
            self.cur.execute(
                "INSERT INTO reliability (entity_table, entity_id, column_name, reliability_value) "
                "VALUES (%s, %s, %s, %s)",
                (table_name, entity_id, column_name, reliability_value)
            )
        except Exception as e:
            print(f"Error adding reliability: {e}")

    def create_entity_identifier(self, entity_id, entity_type, identificator, access_or_meta):
        """Создаем идентификаторы сущностей с поддержкой meta_info"""
        name_info = identificator.get('name', {})
        try:
            # Определяем, переданы ли access_options или meta_info
            if 'url' in access_or_meta or 'external_title' in access_or_meta:
                # Это meta_info
                meta_info = access_or_meta
                source_url = meta_info.get('url')
                external_title = meta_info.get('external_title')
                video_url = meta_info.get('video')
            else:
                # Это access_options (старый формат)
                source_url = access_or_meta.get('source_url')
                external_title = access_or_meta.get('original_title')
                video_url = None

            self.cur.execute(
                "INSERT INTO entity_identifier (url, file_path, name_ru, name_en, name_latin) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (
                    source_url,
                    access_or_meta.get('file_path'),  # Может быть в обоих форматах
                    name_info.get('common') or external_title,
                    name_info.get('en_name'),
                    name_info.get('scientific')
                )
            )
            identifier_id = self.cur.fetchone()[0]
            
            self.cur.execute(
                "INSERT INTO entity_identifier_link (entity_id, entity_type, identifier_id) "
                "VALUES (%s, %s, %s)",
                (entity_id, entity_type, identifier_id)
            )
            
            # Если есть video URL, создаем external_link
            video_url = access_or_meta.get('video')
            if video_url:
                self.cur.execute(
                    "INSERT INTO external_link (url, title, link_type, platform) "
                    "VALUES (%s, %s, %s, %s) RETURNING id",
                    (
                        video_url,
                        f"Видео: {name_info.get('common') or external_title}",
                        'video',
                        self._detect_video_platform(video_url)
                    )
                )
                external_link_id = self.cur.fetchone()[0]
                
                # Связываем external_link с entity_identifier
                self.cur.execute(
                    "INSERT INTO entity_relation (source_id, source_type, target_id, target_type, relation_type) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (identifier_id, 'entity_identifier', external_link_id, 'external_link', 'ссылка на видео')
                )
            
            return identifier_id
        except Exception as e:
            print(f"Error creating entity identifier: {e}")
            return None

    def _detect_video_platform(self, url):
        """Определяет платформу видео по URL"""
        if 'youtube.com' in url or 'youtu.be' in url:
            return 'YouTube'
        elif 'rutube.ru' in url:
            return 'Rutube'
        elif 'vk.com' in url:
            return 'VK'
        elif 'dzen.ru' in url:
            return 'Yandex.Dzen'
        else:
            return 'Other'

    def get_title(self, resource):
        """Получаем заголовок из различных источников с приоритетом"""
        common_name = resource['identificator'].get('name', {}).get('common')
        if common_name:
            return common_name
        
        original_title = resource.get('access_options', {}).get('original_title')
        if original_title:
            return original_title
        
        return resource['identificator'].get('id', 'Без названия')

    def find_biological_entity(self, common_name, scientific_name):
        """Ищем по научному и общеупотребительному имени с кэшированием по обоим"""
        if scientific_name and scientific_name in self.bio_entity_cache:
            return self.bio_entity_cache[scientific_name]
        
        if common_name and common_name in self.bio_entity_cache:
            return self.bio_entity_cache[common_name]
        
        try:
            conditions = []
            params = []
            
            if scientific_name:
                conditions.append("scientific_name = %s")
                params.append(scientific_name)
            if common_name:
                conditions.append("common_name_ru = %s")
                params.append(common_name)

            if conditions:
                query = "SELECT id, scientific_name, common_name_ru FROM biological_entity WHERE "
                query += " OR ".join(conditions)
                self.cur.execute(query, params)
                result = self.cur.fetchone()
                
                if result:
                    bio_id, sci_name, com_name = result
                    if sci_name:
                        self.bio_entity_cache[sci_name] = bio_id
                    if com_name:
                        self.bio_entity_cache[com_name] = bio_id
                    return bio_id
        except Exception as e:
            print(f"Error finding biological entity: {e}")
        return None

    def process_biological_entity(self, source_id, source_type, name_info, classification, feature_data, information_subtype=None):
        """Создаем биологическую сущность и связи с учетом синонимов и типа"""
        try:
            common_name = self.normalize_species_name(name_info.get('common')) or 'Неизвестный вид'
            scientific_name = name_info.get('scientific')
            
            # ОПРЕДЕЛЯЕМ ТИП ИЗ feature_data
            biological_type = information_subtype
            if not biological_type and feature_data:
                biological_type = self.determine_biological_type(feature_data)
            
            bio_id = self.find_biological_entity(common_name, scientific_name)
            
            if not bio_id:
                self.cur.execute(
                    "INSERT INTO biological_entity (common_name_ru, scientific_name, description, type, feature_data) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (
                        common_name,
                        scientific_name,
                        feature_data.get('image_caption'),
                        biological_type,  # ИСПОЛЬЗУЕМ ОПРЕДЕЛЕННЫЙ ТИП
                        Json({
                            'classification': classification,
                            'habitat': feature_data.get('habitat'),
                            'season': feature_data.get('season'),
                            'original_names': [name_info.get('common')],
                            # Сохраняем оригинальные поля для истории
                            'flora_type': feature_data.get('flora_type'),
                            'fauna_type': feature_data.get('fauna_type'),
                            'information_subtype': information_subtype
                        })
                    )
                )
                bio_id = self.cur.fetchone()[0]
                
                self.bio_entity_cache[common_name] = bio_id
                if scientific_name:
                    self.bio_entity_cache[scientific_name] = bio_id
                if name_info.get('common'):
                    self.bio_entity_cache[name_info.get('common')] = bio_id
                    
                self.add_reliability('biological_entity', bio_id, name_info.get('source'))
            else:
                # Если сущность уже существует, обновляем type если он не установлен
                if biological_type:
                    self.cur.execute(
                        "UPDATE biological_entity SET type = %s WHERE id = %s AND type IS NULL",
                        (biological_type, bio_id)
                    )
            
            self.cur.execute(
                "INSERT INTO entity_relation (source_id, source_type, target_id, target_type, relation_type) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                (source_id, source_type, bio_id, 'biological_entity', 'изображение объекта')
            )
            
            return bio_id
            
        except Exception as e:
            print(f"Error processing biological entity: {e}")
            return None

    def process_geographical_data(self, entity_id, entity_type, location, name_info):
        """Обрабатываем географические данные с координатами и создаем map_content"""
        try:
            coords = location.get('coordinates', {})
            lat = self.clean_coordinate(coords.get('latitude'))
            lon = self.clean_coordinate(coords.get('longitude'))
            
            if lat is None or lon is None:
                print(f"Warning: Invalid coordinates for {entity_type} {entity_id}")
                return None
                
            geo_name = location.get('location') or name_info.get('common') or 'Геоточка'
            
            self.cur.execute(
                "SELECT id FROM geographical_entity WHERE name_ru = %s "
                "AND feature_data->'coordinates'->>'latitude' = %s "
                "AND feature_data->'coordinates'->>'longitude' = %s",
                (geo_name, str(lat), str(lon))
            )
            existing_geo = self.cur.fetchone()

            geo_id = None
            if existing_geo:
                geo_id = existing_geo[0]
            else:
                self.cur.execute(
                    "INSERT INTO geographical_entity (name_ru, description, feature_data) "
                    "VALUES (%s, %s, %s) RETURNING id",
                    (
                        geo_name,
                        f"{location.get('region', '')}, {location.get('country', '')}",
                        Json({
                            **location,
                            'coordinates': {
                                'latitude': lat,
                                'longitude': lon
                            }
                        })
                    )
                )
                geo_id = self.cur.fetchone()[0]
                
                self.add_reliability('geographical_entity', geo_id, name_info.get('source'))
                
                self.cur.execute(
                    "INSERT INTO map_content (title, geometry, feature_data) "
                    "VALUES (%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s) RETURNING id",
                    (
                        f"Координаты {geo_name}",
                        lon,
                        lat,
                        Json(location)
                    )
                )
                map_id = self.cur.fetchone()[0]
                
                self.cur.execute(
                    "INSERT INTO entity_geo (entity_id, entity_type, geographical_entity_id) "
                    "VALUES (%s, %s, %s)",
                    (map_id, 'map_content', geo_id)
                )

            self.cur.execute(
                "INSERT INTO entity_geo (entity_id, entity_type, geographical_entity_id) "
                "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (entity_id, entity_type, geo_id)
            )

            return geo_id
            
        except Exception as e:
            print(f"Error processing geographical data for {entity_type} {entity_id}: {e}")
            return None

    def extract_settlements_and_natural_objects(self, resource):
        """
        Извлекает населенные пункты и природные объекты из географического объекта
        Возвращает список словарей с информацией об объектах
        """
        settlements = []
        natural_objects = []
        
        try:
            feature_data = resource.get('feature_data', {})
            location_info = feature_data.get('location_info', {})
            
            # 1. Извлекаем населенный пункт из exact_location
            exact_location = location_info.get('exact_location', '')
            if exact_location:
                settlement = self._parse_settlement_from_location(exact_location)
                if settlement and settlement not in settlements:
                    settlements.append(settlement)
            
            # 2. Извлекаем регион (может содержать населенные пункты)
            region = location_info.get('region', '')
            if region:
                region_settlement = self._parse_settlement_from_region(region)
                if region_settlement and region_settlement not in settlements:
                    settlements.append(region_settlement)
            
            # 3. Извлекаем природные объекты из nearby_places
            nearby_places = location_info.get('nearby_places', [])
            for place in nearby_places:
                if isinstance(place, dict):
                    place_name = place.get('name', '')
                    place_type = place.get('type', '')
                    
                    if self._is_natural_object(place_type):
                        natural_obj = {
                            'name': place_name,
                            'type': place_type,
                            'relation': place.get('relation', ''),
                            'source': 'nearby_places'
                        }
                        if natural_obj not in natural_objects:
                            natural_objects.append(natural_obj)
            
            return {
                'settlements': settlements,
                'natural_objects': natural_objects
            }
            
        except Exception as e:
            print(f"Error extracting settlements and natural objects: {e}")
            return {'settlements': [], 'natural_objects': []}
    
    def _parse_settlement_from_location(self, location_str):
        """Парсит населенный пункт из строки местоположения"""
        if not location_str:
            return None
            
        # Паттерны для извлечения населенных пунктов
        patterns = [
            r'город\s+([^,]+)',  # "город Бодайбо"
            r'г\.\s*([^,]+)',    # "г. Бодайбо"
            r'посёлок\s+([^,]+)', # "посёлок Таксимо"
            r'п\.\s*([^,]+)',     # "п. Таксимо"
            r'село\s+([^,]+)',    # "село Танхой"
            r'с\.\s*([^,]+)',     # "с. Танхой"
            r'деревня\s+([^,]+)', # "деревня Листвянка"
            r'д\.\s*([^,]+)',     # "д. Листвянка"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, location_str.lower())
            if match:
                settlement_name = match.group(1).strip()
                # Определяем тип населенного пункта
                settlement_type = self._determine_settlement_type(pattern)
                return {
                    'name': settlement_name.title(),
                    'type': settlement_type,
                    'source': 'exact_location'
                }
        
        # Если не нашли по паттернам, пробуем взять первое слово до запятой
        if ',' in location_str:
            first_part = location_str.split(',')[0].strip()
            if any(word in first_part.lower() for word in ['город', 'посёлок', 'село', 'деревня']):
                return None  # Пропускаем, если есть указание типа но не распарсилось
            return {
                'name': first_part,
                'type': 'Населенные пункты',
                'source': 'exact_location'
            }
        
        return None
    
    def _parse_settlement_from_region(self, region_str):
        """Парсит населенный пункт из названия региона"""
        if not region_str:
            return None
            
        # Проверяем, является ли регион населенным пунктом
        region_lower = region_str.lower()
        
        # Список суффиксов, указывающих на район (не населенный пункт)
        district_indicators = ['ский район', 'кой район', 'ой район', 'район', 'ский р-н', 'кой р-н']
        
        if any(indicator in region_lower for indicator in district_indicators):
            return None  # Это район, а не населенный пункт
        
        # Если в регионе нет указания на район, возможно это населенный пункт
        return {
            'name': region_str,
            'type': 'Населенные пункты',
            'source': 'region'
        }
    
    def _determine_settlement_type(self, pattern):
        """Определяет тип населенного пункта по паттерну"""
        type_mapping = {
            r'город\s+([^,]+)': 'город',
            r'г\.\s*([^,]+)': 'город',
            r'посёлок\s+([^,]+)': 'посёлок',
            r'п\.\s*([^,]+)': 'посёлок',
            r'село\s+([^,]+)': 'село',
            r'с\.\s*([^,]+)': 'село',
            r'деревня\s+([^,]+)': 'деревня',
            r'д\.\s*([^,]+)': 'деревня'
        }
        
        return type_mapping.get(pattern, 'населенный пункт')
    
    def _is_natural_object(self, object_type):
        """Проверяет, является ли объект природным"""
        natural_types = [
            'река', 'озеро', 'гора', 'хребет', 'лес', 'поле', 'долина',
            'водопад', 'источник', 'бухта', 'залив', 'мыс', 'остров',
            'пещера', 'ущелье', 'каньон', 'плато', 'вулкан'
        ]
        
        return object_type.lower() in natural_types
    
    def check_duplicate_geographical_entity(self, name, entity_type=None):
        """
        Проверяет, существует ли уже географический объект с таким названием
        Возвращает ID если найден, иначе None
        """
        try:
            if entity_type:
                self.cur.execute(
                    "SELECT id FROM geographical_entity WHERE name_ru = %s AND type = %s",
                    (name, entity_type)
                )
            else:
                self.cur.execute(
                    "SELECT id FROM geographical_entity WHERE name_ru = %s",
                    (name,)
                )
            
            result = self.cur.fetchone()
            return result[0] if result else None
            
        except Exception as e:
            print(f"Error checking duplicate for {name}: {e}")
            return None
    

    def create_settlement_entity(self, settlement_info):
        """Создает географическую сущность для населенного пункта с координатами из geodb.json"""
        try:
            name = settlement_info['name']
            settlement_type = settlement_info['type']
            
            # Проверяем дубликат
            existing_id = self.check_duplicate_geographical_entity(name, 'населенный пункт')
            if existing_id:
                print(f"Населенный пункт '{name}' уже существует (id: {existing_id})")
                return existing_id
            
            # Ищем геоданные в geodb.json
            geo_data = self.get_geo_data(name)
            feature_data = {}
            
            if geo_data:
                feature_data = {
                    'source': 'geodb.json',
                    'original_name': name,
                    'geodb_data': geo_data.get('properties', {}),
                    'has_precise_geometry': 'geometry' in geo_data
                }
            
            # Создаем новую сущность
            self.cur.execute(
                "INSERT INTO geographical_entity (name_ru, type, description, feature_data) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (
                    name,
                    'Населенные пункты',
                    f"{settlement_type.capitalize()} {name}",
                    Json(feature_data) if feature_data else None
                )
            )
            settlement_id = self.cur.fetchone()[0]
            
            # Добавляем информацию о достоверности
            self.add_reliability('geographical_entity', settlement_id, settlement_info['source'])
            
            # Создаем map_content с геометрией из geodb.json, если есть
            if geo_data and 'geometry' in geo_data:
                self._create_map_content_for_entity(
                    settlement_id, 
                    'geographical_entity', 
                    name, 
                    geo_data['geometry'],
                    'Населенные пункты'
                )
            else:
                # ВАЖНО: НЕ добавляем в missing_geometry_objects здесь
                print(f"⚠️ Для населенного пункта '{name}' не найдена геометрия в geodb.json")
            
            print(f"Создан населенный пункт: {name} (тип: {settlement_type}, id: {settlement_id})")
            return settlement_id
            
        except Exception as e:
            print(f"Error creating settlement entity for {settlement_info}: {e}")
            return None
    
    def create_natural_entity(self, natural_object_info):
        """Создает географическую сущность для природного объекта с координатами из geodb.json"""
        try:
            name = natural_object_info['name']
            natural_type = natural_object_info['type']
            
            # Проверяем дубликат
            existing_id = self.check_duplicate_geographical_entity(name, natural_type)
            if existing_id:
                print(f"Природный объект '{name}' уже существует (id: {existing_id})")
                return existing_id
            
            # Ищем геоданные в geodb.json
            geo_data = self.get_geo_data(name)
            feature_data = {}
            
            if geo_data:
                feature_data = {
                    'source': 'geodb.json',
                    'original_name': name,
                    'geodb_data': geo_data.get('properties', {}),
                    'has_precise_geometry': 'geometry' in geo_data
                }
            
            # Создаем новую сущность
            self.cur.execute(
                "INSERT INTO geographical_entity (name_ru, type, description, feature_data) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (
                    name,
                    natural_type,
                    f"{natural_type.capitalize()} {name}",
                    Json(feature_data) if feature_data else None
                )
            )
            natural_id = self.cur.fetchone()[0]
            
            # Добавляем информацию о достоверности
            self.add_reliability('geographical_entity', natural_id, natural_object_info['source'])
            
            # Создаем map_content с геометрией из geodb.json, если есть
            if geo_data and 'geometry' in geo_data:
                self._create_map_content_for_entity(
                    natural_id, 
                    'geographical_entity', 
                    name, 
                    geo_data['geometry'],
                    natural_type
                )
            else:
                # ВАЖНО: НЕ добавляем в missing_geometry_objects здесь
                print(f"⚠️ Для природного объекта '{name}' не найдена геометрия в geodb.json")
            
            print(f"Создан природный объект: {name} (тип: {natural_type}, id: {natural_id})")
            return natural_id
            
        except Exception as e:
            print(f"Error creating natural entity for {natural_object_info}: {e}")
            return None

    def _create_map_content_for_entity(self, entity_id, entity_type, name, geometry, obj_type):
        """Создает map_content для сущности с геометрией из geodb.json"""
        try:
            # Проверяем, существует ли уже map_content для этой геометрии
            self.cur.execute(
                """
                SELECT mc.id FROM map_content mc
                WHERE ST_Equals(mc.geometry, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
                LIMIT 1
                """,
                (json.dumps(geometry),)
            )
            existing_map = self.cur.fetchone()
            
            if existing_map:
                map_id = existing_map[0]
                print(f"Map content уже существует для {name} (id: {map_id})")
            else:
                # Создаем новый map_content
                self.cur.execute(
                    """
                    INSERT INTO map_content (title, geometry, feature_data)
                    VALUES (%s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s)
                    RETURNING id
                    """,
                    (
                        f"Геометрия {name}",
                        json.dumps(geometry),
                        Json({
                            'source': 'geodb.json',
                            f'{entity_type}_id': entity_id,
                            'type': obj_type,
                            'original_name': name,
                            'has_precise_geometry': True
                        })
                    )
                )
                map_id = self.cur.fetchone()[0]
                print(f"Создан map_content для {name} (id: {map_id})")
            
            # Связываем map_content с географической сущностью
            self.cur.execute(
                """
                INSERT INTO entity_geo 
                (entity_id, entity_type, geographical_entity_id)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (map_id, 'map_content', entity_id)
            )
            
            return map_id
            
        except Exception as e:
            print(f"Error creating map content for {name}: {e}")
            return None

    
    def _get_natural_object_relation(self, natural_obj):
        """Определяет тип отношения для природного объекта"""
        relation_mapping = {
            'река': 'расположен на берегу',
            'озеро': 'расположен у',
            'гора': 'расположен у подножия',
            'хребет': 'расположен в районе'
        }
        
        return relation_mapping.get(natural_obj['type'], 'расположен рядом с')

    def process_geographical_object(self, resource):
        """Обработка географических объектов с созданием текстового описания и связей"""
        try:
            identificator = resource['identificator']
            name_info = identificator.get('name', {})
            geo_synonyms = resource.get('geo_synonyms', [])
            
            common_name = name_info.get('common')
            geo_entity_type = resource.get('geo_entity_type', 'Географический')
            description = resource.get('description', '')
            coordinates = resource.get('coordinates', {})
            
            # ФИКС: Сохраняем in_stoplist как число
            in_stoplist_value = self.safe_convert_in_stoplist(resource.get('in_stoplist'))
            
            # Создаем базовый feature_data с in_stoplist
            feature_data = {
                'source': 'sights.json',
                'original_name': common_name,
                'coordinates': coordinates,
                'geo_synonyms': geo_synonyms,
                'information_type': resource.get('information_type'),
                'validation_status': resource.get('validation_status'),
                'validation_result': resource.get('validation_result'),
                'baikal_relation': resource.get('baikal_relation'),
                'blacklist_detected': resource.get('blacklist_detected'),
                'blacklist_risk': resource.get('blacklist_risk'),
                'finish_reason': resource.get('finish_reason'),
                'in_stoplist': in_stoplist_value,  # Сохраняем как число
                # ДОБАВЛЯЕМ meta_info
                'meta_info': resource.get('meta_info', {})
            }

            # Добавляем данные из resource['feature_data'], если они есть
            if 'feature_data' in resource and resource['feature_data']:
                import copy
                resource_feature_data = copy.deepcopy(resource['feature_data'])
                feature_data.update(resource_feature_data)

            feature_data_json = Json(feature_data)

            # 1. Создаем основную географическую сущность
            self.cur.execute(
                "INSERT INTO geographical_entity (name_ru, description, type, feature_data) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (
                    common_name,
                    description,
                    geo_entity_type,
                    feature_data_json
                )
            )
            geo_id = self.cur.fetchone()[0]
            entity_type = 'geographical_entity'
            
            self.add_reliability('geographical_entity', geo_id, name_info.get('source'))
            
            text_content_id = self._create_geographical_text_content(
                common_name, 
                description, 
                geo_entity_type,
                coordinates,
                name_info.get('source'),
                resource.get('meta_info')  # Добавляем meta_info
            )
            
            # 3. Создаем связь между текстовым описанием и географической сущностью
            if text_content_id:
                self.cur.execute(
                    "INSERT INTO entity_relation (source_id, source_type, target_id, target_type, relation_type) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (text_content_id, 'text_content', geo_id, entity_type, 'описание объекта')
                )
            
            # 4. Создаем идентификатор с учетом meta_info
            self.create_entity_identifier(geo_id, entity_type, identificator, resource.get('meta_info', {}))
            
            lat = self.clean_coordinate(coordinates.get('latitude'))
            lon = self.clean_coordinate(coordinates.get('longitude'))
            
            # Пробуем найти геометрию в geodb.json
            geo_data = None
            for geo_name in [common_name] + geo_synonyms:
                geo_data = self.get_geo_data(geo_name)
                if geo_data and 'geometry' in geo_data:
                    break
            
            has_geometry = False
            if geo_data and 'geometry' in geo_data:
                # Используем геометрию из geodb.json (приоритет)
                self._create_map_content_for_entity(
                    geo_id,
                    'geographical_entity',
                    common_name,
                    geo_data['geometry'],
                    geo_entity_type
                )
                has_geometry = True
            elif lat is not None and lon is not None:
                # Используем точечные координаты из ресурса (резервный вариант)
                self.cur.execute(
                    "INSERT INTO map_content (title, geometry, feature_data) "
                    "VALUES (%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s) RETURNING id",
                    (
                        f"Координаты {common_name}",
                        lon,
                        lat,
                        Json({
                            'source': 'resource_coordinates', 
                            'geo_entity_id': geo_id,
                            'type': geo_entity_type,
                            'original_name': common_name,
                            'has_precise_geometry': False,
                            'meta_info': resource.get('meta_info', {})  # Добавляем meta_info
                        })
                    )
                )
                map_id = self.cur.fetchone()[0]
                
                # Связываем map_content с географической сущностью
                self.cur.execute(
                    "INSERT INTO entity_geo (entity_id, entity_type, geographical_entity_id) "
                    "VALUES (%s, %s, %s)",
                    (map_id, 'map_content', geo_id)
                )
                has_geometry = True
            
            # 5. Только если нет геометрии вообще - добавляем в missing_geometry_objects
            if not has_geometry:
                self.missing_geometry_objects.add(common_name)
                print(f"⚠️ Для географического объекта '{common_name}' не найдена геометрия")
            
            # 6. Обрабатываем дополнительные geo_synonyms
            for geo_name in geo_synonyms:
                if geo_name and geo_name != common_name:
                    self.process_geo_mention(geo_id, entity_type, geo_name, name_info)
            
            # 7. Извлекаем и создаем населенные пункты и природные объекты
            extracted_objects = self.extract_settlements_and_natural_objects(resource)
            
            # Создаем населенные пункты
            for settlement in extracted_objects['settlements']:
                settlement_id = self.create_settlement_entity(settlement)
                if settlement_id:
                    # Создаем связь между основным объектом и населенным пунктом
                    self.cur.execute(
                        "INSERT INTO entity_relation (source_id, source_type, target_id, target_type, relation_type) "
                        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        (geo_id, 'geographical_entity', settlement_id, 'geographical_entity', 'расположен в')
                    )
            
            # Создаем природные объекты
            for natural_obj in extracted_objects['natural_objects']:
                natural_id = self.create_natural_entity(natural_obj)
                if natural_id:
                    # Создаем связь между основным объектом и природным объектом
                    relation_type = self._get_natural_object_relation(natural_obj)
                    self.cur.execute(
                        "INSERT INTO entity_relation (source_id, source_type, target_id, target_type, relation_type) "
                        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        (geo_id, 'geographical_entity', natural_id, 'geographical_entity', relation_type)
                    )
            
            print(f"✅ Создан географический объект: {common_name} (тип: {geo_entity_type}, id: {geo_id})")
            return geo_id
            
        except Exception as e:
            print(f"❌ Error processing geographical object: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_geographical_text_content(self, name, description, geo_type, coordinates, source, meta_info=None):
        """Создает текстовое описание для географического объекта с эмбеддингом и meta_info"""
        try:
            # Формируем структурированные данные
            structured_data = {
                "geographical_info": {
                    "object_type": geo_type,
                    "coordinates": coordinates,
                    "name": name,
                    "description": description
                },
                "metadata": {
                    "source": source,
                    "import_timestamp": datetime.now().isoformat(),
                    "meta_info": meta_info or {}  # Добавляем meta_info
                }
            }
            
            # Генерируем эмбеддинг из названия и описания
            text_for_embedding = f"{name}. {description}"
            embedding = self.generate_embedding(text_for_embedding)
            
            # Вставляем в text_content
            self.cur.execute(
                "INSERT INTO text_content (title, content, structured_data, description, embedding) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (
                    name,
                    description,  # Основной контент
                    Json(structured_data),
                    f"Описание географического объекта: {geo_type}",
                    embedding
                )
            )
            text_id = self.cur.fetchone()[0]
            
            # Добавляем информацию о достоверности
            self.add_reliability('text_content', text_id, source)
            
            # Создаем идентификатор для текстового контента
            self.cur.execute(
                "INSERT INTO entity_identifier (name_ru) VALUES (%s) RETURNING id",
                (f"Текстовое описание: {name}",)
            )
            ident_id = self.cur.fetchone()[0]
            
            self.cur.execute(
                "INSERT INTO entity_identifier_link (entity_id, entity_type, identifier_id) "
                "VALUES (%s, %s, %s)",
                (text_id, 'text_content', ident_id)
            )
            
            print(f"Создано текстовое описание для географического объекта: {name} (id: {text_id})")
            return text_id
            
        except Exception as e:
            print(f"Error creating geographical text content: {e}")
            return None
    def get_text_for_embedding(self, resource):
            """Собирает текст для генерации эмбеддинга из всех доступных полей"""
            title = self.get_title(resource)
            structured_data = resource.get('structured_data')
            content = resource.get('content', '')
            
            text_parts = []
            
            # Всегда добавляем заголовок
            if title:
                text_parts.append(title)
            
            # Обрабатываем structured_data - извлекаем все текстовые значения
            if structured_data:
                # Рекурсивно собираем все строковые значения из structured_data
                def extract_text_values(data):
                    if isinstance(data, dict):
                        return ' '.join(extract_text_values(value) for value in data.values())
                    elif isinstance(data, list):
                        return ' '.join(extract_text_values(item) for item in data)
                    elif isinstance(data, str):
                        return data
                    else:
                        return ''
                
                structured_text = extract_text_values(structured_data).strip()
                if structured_text:
                    text_parts.append(structured_text)
            
            # Добавляем обычный контент, если нет structured_data
            elif content:
                text_parts.append(content)
            
            # Объединяем все части
            combined_text = ' '.join(text_parts).strip()
            
            # Логируем для отладки (можно убрать в продакшене)
            print(f"Text for embedding: {combined_text[:200]}...")
            
            return combined_text        
    def process_text(self, resource):
        """Обработка текстовых ресурсов с генерацией эмбеддингов и structured_data"""
        try:
            identificator = resource['identificator']
            access = resource.get('access_options', {})
            name_info = identificator.get('name', {})
            
            title = self.get_title(resource)
            structured_data = resource.get('structured_data')
            
            # ФИКС: Сохраняем in_stoplist как число
            in_stoplist_value = self.safe_convert_in_stoplist(resource.get('in_stoplist'))
            
            # Собираем feature_data для текстового контента
            feature_data = {
                'in_stoplist': in_stoplist_value,  # Сохраняем как число
                'information_type': resource.get('information_type'),
                'source': name_info.get('source')
            }
            
            # Добавляем дополнительные поля, если они есть
            if 'validation_status' in resource:
                feature_data['validation_status'] = resource.get('validation_status')
            if 'validation_result' in resource:
                feature_data['validation_result'] = resource.get('validation_result')
            
            feature_data_json = Json(feature_data) if feature_data else None
            
            # Логируем полученные данные для отладки
            print(f"Processing text: {title}")
            print(f"Has structured_data: {structured_data is not None}")
            print(f"in_stoplist: {in_stoplist_value}")
            
            combined_text = self.get_text_for_embedding(resource)
            print(f"📝 Текст для эмбеддинга: {combined_text[:100]}...")
            
            embedding = self.generate_embedding(combined_text)
            
            if embedding is None:
                print("❌ Не удалось сгенерировать эмбеддинг")
            else:
                print(f"✅ Эмбеддинг сгенерирован, размер: {len(embedding)}")
            
            # Обрабатываем structured_data с проверкой ошибок
            structured_data_json = None
            if structured_data:
                try:
                    structured_data_json = Json(structured_data)
                    print("Structured data processed successfully")
                except Exception as e:
                    print(f"Error converting structured_data to JSON: {e}")
                    # Пытаемся сохранить как строку для отладки
                    try:
                        structured_data_json = Json({"error": f"Failed to parse: {str(structured_data)[:100]}..."})
                    except:
                        structured_data_json = None
            
            # Вставляем данные в базу - content только если нет structured_data
            self.cur.execute(
                "INSERT INTO text_content (title, content, structured_data, description, feature_data, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    title,
                    None if structured_data else resource.get('content', ''),  # content только если нет structured_data
                    structured_data_json,
                    resource.get('brief_annotation', ''),
                    feature_data_json,
                    embedding  
                )
            )
            text_id = self.cur.fetchone()[0]
            entity_type = 'text_content'
            
            self.add_reliability('text_content', text_id, name_info.get('source'))
            
            self.create_entity_identifier(text_id, entity_type, identificator, access)
            
            # Обработка автора
            author_name = access.get('author')
            if author_name:
                author_id = self.get_or_create_author(author_name)
                if author_id:
                    self.cur.execute(
                        "INSERT INTO entity_author (entity_id, entity_type, author_id) "
                        "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (text_id, entity_type, author_id)
                    )
            
            # Обработка географических упоминаний
            geo_synonyms = resource.get('geo_synonyms', [])
            for geo_name in geo_synonyms:
                if geo_name:  # Проверяем, что имя не пустое
                    self.process_geo_mention(text_id, entity_type, geo_name, name_info)
            
            # Обработка биологических сущностей
            if resource.get('information_type') == "Объект флоры и фауны":
                common_name = name_info.get('common')
                scientific_name = name_info.get('scientific')
                
                if common_name or scientific_name:
                    bio_id = self.find_biological_entity(common_name, scientific_name)
                    
                    if not bio_id:
                        information_subtype = resource.get('information_subtype')
                        feature_data = resource.get('feature_data', {})
                        
                        # Создаем новую биологическую сущность
                        self.cur.execute(
                            "INSERT INTO biological_entity (common_name_ru, scientific_name, description, type, feature_data) "
                            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                            (
                                common_name, 
                                scientific_name, 
                                f"Автоматически создано из текста: {title}",
                                information_subtype or self.determine_biological_type(feature_data),  # ОПРЕДЕЛЯЕМ ТИП
                                Json({
                                    'in_stoplist': in_stoplist_value,
                                    'information_subtype': information_subtype,
                                    'flora_type': feature_data.get('flora_type'),
                                    'fauna_type': feature_data.get('fauna_type')
                                })
                            )
                        )
                        bio_id = self.cur.fetchone()[0]
                        
                        # Обновляем кэш
                        if common_name:
                            self.bio_entity_cache[common_name] = bio_id
                        if scientific_name:
                            self.bio_entity_cache[scientific_name] = bio_id
                        
                        self.add_reliability('biological_entity', bio_id, name_info.get('source'))
                    
                    # Создаем связь
                    self.cur.execute(
                        "INSERT INTO entity_relation (source_id, source_type, target_id, target_type, relation_type) "
                        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        (text_id, entity_type, bio_id, 'biological_entity', 'описание объекта')
                    )
            
            print(f"Successfully processed text ID: {text_id}, in_stoplist: {in_stoplist_value}")
            return text_id
            
        except Exception as e:
            print(f"Error processing text: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def process_image(self, resource):
        """Обработка изображений с созданием географических сущностей"""
        try:
            identificator = resource['identificator']
            access = resource.get('access_options', {})
            feature_photo = resource.get('featurePhoto', {})
            name_info = identificator.get('name', {})
            
            title = self.get_title(resource)
            
            # ФИКС: Сохраняем in_stoplist как число
            in_stoplist_value = self.safe_convert_in_stoplist(resource.get('in_stoplist'))
            
            # Добавляем in_stoplist в feature_data изображения
            image_feature_data = feature_photo.copy() if feature_photo else {}
            image_feature_data['in_stoplist'] = in_stoplist_value  # Сохраняем как число
            
            self.cur.execute(
                "INSERT INTO image_content (title, description, feature_data) "
                "VALUES (%s, %s, %s) RETURNING id",
                (
                    title,
                    feature_photo.get('image_caption'),
                    Json(image_feature_data)
                )
            )
            image_id = self.cur.fetchone()[0]
            entity_type = 'image_content'
            
            self.add_reliability('image_content', image_id, name_info.get('source'))
            self.create_entity_identifier(image_id, entity_type, identificator, access)
            
            # ... остальной код метода без изменений
            
            author_name = access.get('author')
            if author_name:
                author_id = self.get_or_create_author(author_name)
                if author_id:
                    self.cur.execute(
                        "INSERT INTO entity_author (entity_id, entity_type, author_id) "
                        "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (image_id, entity_type, author_id)
                    )
            
            date_taken = feature_photo.get('date')
            if date_taken:
                parsed_date = self.parse_date(date_taken)
                if parsed_date:
                    self.cur.execute(
                        "INSERT INTO temporal_reference (resource_creation_date) "
                        "VALUES (%s) RETURNING id",
                        (parsed_date,)
                    )
                    temporal_id = self.cur.fetchone()[0]
                    self.cur.execute(
                        "INSERT INTO entity_temporal (entity_id, entity_type, temporal_id) "
                        "VALUES (%s, %s, %s)",
                        (image_id, entity_type, temporal_id)
                    )
            
            classification = feature_photo.get('classification_info')
            if classification:
                information_subtype = resource.get('information_subtype')
                self.process_biological_entity(
                    image_id, 
                    entity_type,
                    name_info,
                    classification,
                    feature_photo,
                    information_subtype
                )
            
            location = feature_photo.get('location', {})
            if location:
                self.process_geographical_data(
                    image_id, 
                    entity_type,
                    location,
                    name_info
                )
                
            return image_id
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def process_weather(self, entity_id, entity_type, weather_conditions):
        """Обрабатываем погодные условия"""
        try:
            windy = 'ветер' in weather_conditions.lower()
            rain = 'дождь' in weather_conditions.lower()
            
            self.cur.execute(
                "INSERT INTO weather_reference (weather_conditions, windy, rain) "
                "VALUES (%s, %s, %s) RETURNING id",
                (weather_conditions, windy, rain)
            )
            weather_id = self.cur.fetchone()[0]
            
            self.cur.execute(
                "INSERT INTO entity_weather (entity_id, entity_type, weather_id) "
                "VALUES (%s, %s, %s)",
                (entity_id, entity_type, weather_id)
            )
            
            return weather_id
            
        except Exception as e:
            print(f"Error processing weather: {e}")
            return None
        
    def normalize_geo_name(self, name):
        """Унифицируем регистр названий"""
        if not name:
            return name
        return name.strip().lower()

    def _get_biological_name_from_map(self, resource):
        """Получает название биологической сущности для карт с приоритетом plant_russian_name"""
        # Приоритет 1: plant_russian_name (если есть и валидно)
        plant_russian_name = resource.get('plant_russian_name')
        if plant_russian_name and plant_russian_name.strip():
            return plant_russian_name.strip()
        
        # Приоритет 2: из common name (убираем "Место обитания")
        common_name = resource['identificator'].get('name', {}).get('common', '')
        if common_name:
            # Убираем "Место обитания" и лишние пробелы
            cleaned_name = common_name.replace('Место обитания', '').replace('место обитания', '').strip()
            if cleaned_name:
                return cleaned_name
        
        # Приоритет 3: из ID (убираем GEO_)
        resource_id = resource['identificator'].get('id', '')
        if resource_id.startswith('GEO_'):
            return resource_id.replace('GEO_', '').replace('_', ' ').strip()
        
        return 'Неизвестный вид'
    
    def determine_biological_type(self, feature_data):
        """Определяет тип биологического объекта на основе flora_type/fauna_type"""
        if not feature_data:
            return None
        
        # Проверяем fauna_type (приоритет первый)
        fauna_type = feature_data.get('fauna_type')
        if fauna_type and fauna_type.strip():  # Проверяем что не пустая строка
            return "Объект фауны"
        
        # Проверяем flora_type
        flora_type = feature_data.get('flora_type')
        if flora_type and flora_type.strip():  # Проверяем что не пустая строка
            return "Объект флоры"
        
        # Проверяем information_subtype (резервный вариант)
        information_subtype = feature_data.get('information_subtype')
        if information_subtype and information_subtype.strip():
            return information_subtype
        
        return None

    def process_map(self, resource):
        identificator = resource['identificator']
        name_info = identificator.get('name', {})
        geo_synonyms = resource.get('geo_synonyms', [])
        
        common_name = self._get_biological_name_from_map(resource)
        
        information_subtype = resource.get('information_subtype')
        feature_data = resource.get('feature_data', {})
        bio_id = self._process_biological_entity(
            common_name,
            resource.get('plant_latin_name'),
            name_info.get('source'),
            resource.get('in_stoplist', False),
            information_subtype,
            feature_data
        )

        # Обрабатываем все географические объекты
        for geo_name in geo_synonyms:
            if not geo_name:
                continue
                
            # Получаем упрощенное название
            simplified_name = self.simplify_geo_name(geo_name)
            
            # Ищем геометрию по упрощенному названию
            geo_data = self.get_geo_data(simplified_name)
            
            if geo_data and 'geometry' in geo_data:
                # Создаем географическую сущность для полного названия
                full_geo_id = self.process_geo_mention(None, None, geo_name, name_info)
                
                # Проверяем, существует ли уже map_content для этой геометрии
                self.cur.execute(
                    """
                    SELECT mc.id FROM map_content mc
                    WHERE ST_Equals(mc.geometry, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
                    LIMIT 1
                    """,
                    (json.dumps(geo_data['geometry']),)
                )
                existing_map = self.cur.fetchone()
                
                if existing_map:
                    map_id = existing_map[0]
                else:
                    # Создаем map_content только если он не существует
                    self.cur.execute(
                        """
                        INSERT INTO map_content (title, geometry, feature_data)
                        VALUES (%s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s)
                        RETURNING id
                        """,
                        (
                            f"Карта: {simplified_name}",
                            json.dumps(geo_data['geometry']),
                            Json({
                                'source': 'geodb.json',
                                'original_name': simplified_name,
                                'full_name': geo_name
                            })
                        )
                    )
                    map_id = self.cur.fetchone()[0]
                
                # Связываем map_content с географической сущностью
                self.cur.execute(
                    """
                    INSERT INTO entity_geo 
                    (entity_id, entity_type, geographical_entity_id)
                    VALUES (%s, 'map_content', %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (map_id, full_geo_id)
                )
                
                # Связываем биологическую сущность с географической
                if bio_id:
                    self.cur.execute(
                        """
                        INSERT INTO entity_geo 
                        (entity_id, entity_type, geographical_entity_id)
                        VALUES (%s, 'biological_entity', %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (bio_id, full_geo_id)
                    )

        return bio_id
    
    def _process_biological_entity(self, common_name, scientific_name, source, in_stoplist_value=None, information_subtype=None, feature_data=None):
        """Вспомогательный метод для обработки биологической сущности с типом"""
        if not common_name and not scientific_name:
            return None
        
        # Нормализуем названия
        if common_name:
            common_name = self.normalize_species_name(common_name)
        
        bio_id = self.find_biological_entity(common_name, scientific_name)
        
        # ОПРЕДЕЛЯЕМ ТИП ИЗ feature_data
        biological_type = information_subtype
        if not biological_type and feature_data:
            biological_type = self.determine_biological_type(feature_data)
        
        feature_data_dict = {}
        if in_stoplist_value is not None:
            feature_data_dict['in_stoplist'] = self.safe_convert_in_stoplist(in_stoplist_value)
        if feature_data:
            # Сохраняем оригинальные поля
            feature_data_dict.update({
                'flora_type': feature_data.get('flora_type'),
                'fauna_type': feature_data.get('fauna_type'),
                'information_subtype': information_subtype
            })
        
        if bio_id:
            # Если сущность уже существует, обновляем type если он не установлен
            if biological_type:
                self.cur.execute(
                    "UPDATE biological_entity SET type = %s WHERE id = %s AND type IS NULL",
                    (biological_type, bio_id)
                )
            return bio_id
            
        # Создаем новую биологическую сущность
        self.cur.execute(
            """
            INSERT INTO biological_entity 
            (common_name_ru, scientific_name, type, feature_data) 
            VALUES (%s, %s, %s, %s) 
            RETURNING id
            """,
            (common_name, scientific_name, biological_type, Json(feature_data_dict) if feature_data_dict else None)
        )
        bio_id = self.cur.fetchone()[0]
        
        # Обновляем кэш
        for name in filter(None, [common_name, scientific_name]):
            self.bio_entity_cache[name] = bio_id
            
        self.add_reliability('biological_entity', bio_id, source)
        
        # Создаем идентификатор
        self.cur.execute(
            """
            INSERT INTO entity_identifier 
            (name_ru, name_latin) 
            VALUES (%s, %s) 
            RETURNING id
            """,
            (common_name, scientific_name)
        )
        ident_id = self.cur.fetchone()[0]
        
        self.cur.execute(
            """
            INSERT INTO entity_identifier_link
            (entity_id, entity_type, identifier_id)
            VALUES (%s, 'biological_entity', %s)
            """,
            (bio_id, ident_id)
        )
        
        return bio_id

    def import_resources(self, json_file):
        """Основной метод импорта с улучшенным управлением транзакциями"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        success_count = 0
        error_count = 0
        
        for i, resource in enumerate(data['resources'], 1):
            try:
                print(f"\nProcessing resource {i}/{len(data['resources'])}: {resource.get('type')}")
                
                rtype = resource['type']
                if rtype == 'Изображение':
                    result = self.process_image(resource)
                elif rtype == 'Текст':
                    result = self.process_text(resource)
                elif rtype == 'Картографическая информация':
                    result = self.process_map(resource)
                elif rtype == 'Географический объект':
                    result = self.process_geographical_object(resource)
                else:
                    print(f"Unknown resource type: {rtype}")
                    result = None
                
                if result:
                    self.conn.commit()
                    success_count += 1
                else:
                    self.conn.rollback()
                    error_count += 1
                
            except Exception as e:
                print(f"Error processing resource {i}: {e}")
                import traceback
                traceback.print_exc()
                self.conn.rollback()
                error_count += 1
                # Сброс кэшей при ошибке
                self.entity_cache = {}
                self.author_cache = {}
                self.bio_entity_cache = {}

        print(f"\nImport completed. Success: {success_count}, Errors: {error_count}")
            
    def run(self, json_file):
        try:
            self.connect()
            self.import_resources(json_file)
            self.save_missing_geometry_objects()
            print("Импорт успешно завершен!")
        except Exception as e:
            print(f"Ошибка импорта: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.disconnect()
            
if __name__ == "__main__":
    importer = NewResourceImporter()
    try:
        importer.connect()
        importer.import_resources("../../json_files/resources_dist.json")
        importer.save_missing_geometry_objects()
    finally:
        importer.disconnect()