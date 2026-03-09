import os
import json
import sys
import time
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
from dotenv import load_dotenv
import re
from datetime import datetime
from pathlib import Path
import argparse

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class NewResourceImporter:
    def __init__(self, incremental_mode=False):
        self.db_config = {
            "dbname": os.getenv("DB_NAME", "eco"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "Fdf78yh0a4b!"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }
        self.missing_geometry_objects = set()
        self.incremental_mode = incremental_mode
        
        self.conn = None
        self.cur = None
        self.entity_cache = {}
        self.author_cache = {}
        self.bio_entity_cache = {}
        self.geodb_data = self.load_geodb()
        self.species_synonyms_path = self._get_species_synonyms_path()
        self.species_synonyms = self.load_species_synonyms() or {}
        
    def log_progress(self, current, total, resource_type="ресурсов"):
        """Логирование прогресса обработки с принудительным сбросом буфера"""
        progress_percent = (current / total) * 100 if total > 0 else 0
        progress_bar = "█" * int(progress_percent / 5) + "░" * (20 - int(progress_percent / 5))
        
        import sys
        message = f"[{progress_bar}] {current}/{total} {resource_type} ({progress_percent:.1f}%)\n"
        sys.stdout.write(message)
        sys.stdout.flush()
    
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
        geo_name_lower = geo_name.lower()
        
        if ',' in geo_name:
            parts = [part.strip() for part in geo_name.split(',')]
            for part in reversed(parts):
                if part and part in self.geodb_data:
                    return self.geodb_data[part]
                for name, data in self.geodb_data.items():
                    if name.lower() == part.lower():
                        return data
        
        geo_words = set(geo_name_lower.split())
        best_match = None
        best_score = 0
        
        for name, data in self.geodb_data.items():
            name_lower = name.lower()
            name_words = set(name_lower.split())
            
            common_words = geo_words.intersection(name_words)
            score = len(common_words)
            
            if score > best_score:
                best_score = score
                best_match = data
        
        if best_score >= 2:
            return best_match
        
        return None
    
    def process_geo_mention(self, source_id, source_type, geo_name, name_info):
        if not geo_name:
            return None
            
        try:
            normalized_name = self.normalize_geo_name(geo_name)
            
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
        
        if ',' in geo_name:
            parts = [part.strip() for part in geo_name.split(',')]
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
        """Определяем путь к файлу object_synonyms.json"""
        current_dir = Path(__file__).parent
        base_dir = current_dir.parent.parent
        json_files_dir = base_dir / "json_files"
        return json_files_dir / "object_synonyms.json"
    
    def load_species_synonyms(self):
        """Загрузка синонимов видов из JSON-файла"""
        try:
            with open(self.species_synonyms_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'biological_entity' in data:
                return data['biological_entity']
            else:
                print(f"⚠️ В файле {self.species_synonyms_path} нет раздела 'biological_entity'")
                return {}
                
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
            main_name_lower = main_name.lower()
            
            if name_lower == main_name_lower:
                return main_name
            
            for synonym in synonyms:
                if name_lower == synonym.lower():
                    return main_name 
        
        return name
    
    def parse_date(self, date_str):
        """Преобразование даты в формат PostgreSQL"""
        if not date_str:
            return None
            
        try:
            date_str = re.sub(r'[·•]', ' ', date_str).strip()
            
            formats = [
                '%d.%m.%Y %H:%M',
                '%d.%m.%Y',
                '%d.%m.%y %H:%M',
                '%d.%m.%y',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y %H:%M',
                '%d/%m/%Y',
                '%d %m %Y %H:%M',
                '%d %m %Y'
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
            author_id = self.author_cache[cache_key]
            self.cur.execute("SELECT 1 FROM author WHERE id = %s", (author_id,))
            if self.cur.fetchone():
                return author_id
            else:
                del self.author_cache[cache_key]
        
        try:
            self.cur.execute(
                "SELECT id FROM author WHERE full_name = %s AND organization = %s",
                (full_name, organization)
            )
            author = self.cur.fetchone()
            
            if author:
                self.author_cache[cache_key] = author[0]
                return author[0]
                
            self.cur.execute(
                "INSERT INTO author (full_name, organization) VALUES (%s, %s) RETURNING id",
                (full_name, organization)
            )
            author_id = self.cur.fetchone()[0]
            self.conn.commit()
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
            if 'url' in access_or_meta or 'external_title' in access_or_meta:
                meta_info = access_or_meta
                source_url = meta_info.get('url')
                external_title = meta_info.get('external_title')
                video_url = meta_info.get('video')
            else:
                source_url = access_or_meta.get('source_url')
                external_title = access_or_meta.get('original_title')
                video_url = None

            self.cur.execute(
                "INSERT INTO entity_identifier (url, file_path, name_ru, name_en, name_latin) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (
                    source_url,
                    access_or_meta.get('file_path'),
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
                        biological_type,
                        Json({
                            'classification': classification,
                            'habitat': feature_data.get('habitat'),
                            'season': feature_data.get('season'),
                            'original_names': [name_info.get('common')],
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
                if biological_type:
                    self.cur.execute(
                        "UPDATE biological_entity SET type = %s WHERE id = %s AND type IS NULL",
                        (biological_type, bio_id)
                    )
            
            relation_type_map = {
                'image_content': 'изображение объекта',
                'text_content': 'описание объекта'
            }
            
            relation_type = relation_type_map.get(source_type, 'связан с')
            
            self.cur.execute(
                "INSERT INTO entity_relation (source_id, source_type, target_id, target_type, relation_type) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                (source_id, source_type, bio_id, 'biological_entity', relation_type)
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
        """Извлекает населенные пункты и природные объекты из географического объекта"""
        settlements = []
        natural_objects = []
        
        try:
            feature_data = resource.get('feature_data', {})
            location_info = feature_data.get('location_info', {})
            
            exact_location = location_info.get('exact_location', '')
            if exact_location:
                settlement = self._parse_settlement_from_location(exact_location)
                if settlement and settlement not in settlements:
                    settlements.append(settlement)
            
            region = location_info.get('region', '')
            if region:
                region_settlement = self._parse_settlement_from_region(region)
                if region_settlement and region_settlement not in settlements:
                    settlements.append(region_settlement)
            
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
            
        patterns = [
            r'город\s+([^,]+)',
            r'г\.\s*([^,]+)',
            r'посёлок\s+([^,]+)',
            r'п\.\s*([^,]+)',
            r'село\s+([^,]+)',
            r'с\.\s*([^,]+)',
            r'деревня\s+([^,]+)',
            r'д\.\s*([^,]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, location_str.lower())
            if match:
                settlement_name = match.group(1).strip()
                settlement_type = self._determine_settlement_type(pattern)
                return {
                    'name': settlement_name.title(),
                    'type': settlement_type,
                    'source': 'exact_location'
                }
        
        if ',' in location_str:
            first_part = location_str.split(',')[0].strip()
            if any(word in first_part.lower() for word in ['город', 'посёлок', 'село', 'деревня']):
                return None
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
            
        region_lower = region_str.lower()
        
        district_indicators = ['ский район', 'кой район', 'ой район', 'район', 'ский р-н', 'кой р-н']
        
        if any(indicator in region_lower for indicator in district_indicators):
            return None
        
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
        """Проверяет, существует ли уже географический объект с таким названием"""
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
            
            existing_id = self.check_duplicate_geographical_entity(name, 'населенный пункт')
            if existing_id:
                print(f"Населенный пункт '{name}' уже существует (id: {existing_id})")
                return existing_id
            
            geo_data = self.get_geo_data(name)
            feature_data = {}
            
            if geo_data:
                feature_data = {
                    'source': 'geodb.json',
                    'original_name': name,
                    'geodb_data': geo_data.get('properties', {}),
                    'has_precise_geometry': 'geometry' in geo_data
                }
            
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
            
            self.add_reliability('geographical_entity', settlement_id, settlement_info['source'])
            
            if geo_data and 'geometry' in geo_data:
                self._create_map_content_for_entity(
                    settlement_id, 
                    'geographical_entity', 
                    name, 
                    geo_data['geometry'],
                    'Населенные пункты'
                )
            else:
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
            
            existing_id = self.check_duplicate_geographical_entity(name, natural_type)
            if existing_id:
                print(f"Природный объект '{name}' уже существует (id: {existing_id})")
                return existing_id
            
            geo_data = self.get_geo_data(name)
            feature_data = {}
            
            if geo_data:
                feature_data = {
                    'source': 'geodb.json',
                    'original_name': name,
                    'geodb_data': geo_data.get('properties', {}),
                    'has_precise_geometry': 'geometry' in geo_data
                }
            
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
            
            self.add_reliability('geographical_entity', natural_id, natural_object_info['source'])
            
            if geo_data and 'geometry' in geo_data:
                self._create_map_content_for_entity(
                    natural_id, 
                    'geographical_entity', 
                    name, 
                    geo_data['geometry'],
                    natural_type
                )
            else:
                print(f"⚠️ Для природного объекта '{name}' не найдена геометрия в geodb.json")
            
            print(f"Создан природный объект: {name} (тип: {natural_type}, id: {natural_id})")
            return natural_id
            
        except Exception as e:
            print(f"Error creating natural entity for {natural_object_info}: {e}")
            return None

    def _create_map_content_for_entity(self, entity_id, entity_type, name, geometry, obj_type):
        """Создает map_content для сущности с геометрией из geodb.json"""
        try:
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
            
            in_stoplist_value = self.safe_convert_in_stoplist(resource.get('in_stoplist'))
            
            baikal_relation = resource.get('baikal_relation')
            if isinstance(baikal_relation, str):
                baikal_relation = [baikal_relation]
            elif baikal_relation is None:
                baikal_relation = []
            
            feature_data = {
                'source': 'sights.json',
                'original_name': common_name,
                'coordinates': coordinates,
                'geo_synonyms': geo_synonyms,
                'information_type': resource.get('information_type'),
                'validation_status': resource.get('validation_status'),
                'validation_result': resource.get('validation_result'),
                'baikal_relation': baikal_relation,
                'blacklist_detected': resource.get('blacklist_detected'),
                'blacklist_risk': resource.get('blacklist_risk'),
                'finish_reason': resource.get('finish_reason'),
                'in_stoplist': in_stoplist_value,
                'meta_info': resource.get('meta_info', {})
            }
            
            if 'feature_data' in resource and resource['feature_data']:
                import copy
                resource_feature_data = copy.deepcopy(resource['feature_data'])
                feature_data.update(resource_feature_data)

            feature_data_json = Json(feature_data)

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
                resource.get('meta_info')
            )
            
            if text_content_id:
                self.cur.execute(
                    "INSERT INTO entity_relation (source_id, source_type, target_id, target_type, relation_type) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (text_content_id, 'text_content', geo_id, entity_type, 'описание объекта')
                )
            
            self.create_entity_identifier(geo_id, entity_type, identificator, resource.get('meta_info', {}))
            
            lat = self.clean_coordinate(coordinates.get('latitude'))
            lon = self.clean_coordinate(coordinates.get('longitude'))
            
            geo_data = None
            for geo_name in [common_name] + geo_synonyms:
                geo_data = self.get_geo_data(geo_name)
                if geo_data and 'geometry' in geo_data:
                    break
            
            has_geometry = False
            if geo_data and 'geometry' in geo_data:
                self._create_map_content_for_entity(
                    geo_id,
                    'geographical_entity',
                    common_name,
                    geo_data['geometry'],
                    geo_entity_type
                )
                has_geometry = True
            elif lat is not None and lon is not None:
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
                            'meta_info': resource.get('meta_info', {})
                        })
                    )
                )
                map_id = self.cur.fetchone()[0]
                
                self.cur.execute(
                    "INSERT INTO entity_geo (entity_id, entity_type, geographical_entity_id) "
                    "VALUES (%s, %s, %s)",
                    (map_id, 'map_content', geo_id)
                )
                has_geometry = True
            
            if not has_geometry:
                self.missing_geometry_objects.add(common_name)
                print(f"⚠️ Для географического объекта '{common_name}' не найдена геометрия")
            
            for geo_name in geo_synonyms:
                if geo_name and geo_name != common_name:
                    self.process_geo_mention(geo_id, entity_type, geo_name, name_info)
            
            extracted_objects = self.extract_settlements_and_natural_objects(resource)
            
            for settlement in extracted_objects['settlements']:
                settlement_id = self.create_settlement_entity(settlement)
                if settlement_id:
                    self.cur.execute(
                        "INSERT INTO entity_relation (source_id, source_type, target_id, target_type, relation_type) "
                        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        (geo_id, 'geographical_entity', settlement_id, 'geographical_entity', 'расположен в')
                    )
            
            for natural_obj in extracted_objects['natural_objects']:
                natural_id = self.create_natural_entity(natural_obj)
                if natural_id:
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
        """Создает текстовое описание для географического объекта"""
        try:
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
                    "meta_info": meta_info or {}
                }
            }
            
            self.cur.execute(
                "INSERT INTO text_content (title, content, structured_data, description) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (
                    name,
                    description,
                    Json(structured_data),
                    f"Описание географического объекта: {geo_type}"
                )
            )
            text_id = self.cur.fetchone()[0]
            
            self.add_reliability('text_content', text_id, source)
            
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
            
    def process_text(self, resource):
        """Обработка текстовых ресурсов"""
        try:
            identificator = resource['identificator']
            access = resource.get('access_options', {})
            name_info = identificator.get('name', {})
            
            title = self.get_title(resource)
            structured_data = resource.get('structured_data')
            
            in_stoplist_value = self.safe_convert_in_stoplist(resource.get('in_stoplist'))
            
            baikal_relation = resource.get('baikal_relation')
            if isinstance(baikal_relation, str):
                baikal_relation = [baikal_relation]
            elif baikal_relation is None:
                baikal_relation = []
            
            feature_data = {
                'in_stoplist': in_stoplist_value,
                'information_type': resource.get('information_type'),
                'source': name_info.get('source'),
                'baikal_relation': baikal_relation
            }
            
            if 'validation_status' in resource:
                feature_data['validation_status'] = resource.get('validation_status')
            if 'validation_result' in resource:
                feature_data['validation_result'] = resource.get('validation_result')
            
            feature_data_json = Json(feature_data) if feature_data else None
            
            print(f"Processing text: {title}")
            print(f"Has structured_data: {structured_data is not None}")
            print(f"in_stoplist: {in_stoplist_value}")
            
            structured_data_json = None
            if structured_data:
                try:
                    structured_data_json = Json(structured_data)
                    print("Structured data processed successfully")
                except Exception as e:
                    print(f"Error converting structured_data to JSON: {e}")
                    structured_data_json = None
            
            self.cur.execute(
                "INSERT INTO text_content (title, content, structured_data, description, feature_data) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (
                    title,
                    None if structured_data else resource.get('content', ''),
                    structured_data_json,
                    resource.get('brief_annotation', ''),
                    feature_data_json
                )
            )
            text_id = self.cur.fetchone()[0]
            entity_type = 'text_content'
            
            self.add_reliability('text_content', text_id, name_info.get('source'))
            
            self.create_entity_identifier(text_id, entity_type, identificator, access)
            
            author_name = access.get('author')
            if author_name:
                author_id = self.get_or_create_author(author_name)
                if author_id:
                    self.cur.execute(
                        "INSERT INTO entity_author (entity_id, entity_type, author_id) "
                        "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (text_id, entity_type, author_id)
                    )
            
            geo_synonyms = resource.get('geo_synonyms', [])
            for geo_name in geo_synonyms:
                if geo_name:
                    self.process_geo_mention(text_id, entity_type, geo_name, name_info)
            
            bio_id = None
            if resource.get('information_type') == "Объект флоры и фауны":
                common_name = name_info.get('common')
                scientific_name = name_info.get('scientific')
                
                if common_name or scientific_name:
                    information_subtype = resource.get('information_subtype')
                    feature_data_for_bio = resource.get('feature_data', {})
                    
                    bio_id = self.process_biological_entity(
                        text_id,
                        'text_content',
                        name_info,
                        {},
                        feature_data_for_bio,
                        information_subtype
                    )
                    
                    if bio_id:
                        self.cur.execute(
                            "INSERT INTO entity_relation (source_id, source_type, target_id, target_type, relation_type) "
                            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                            (text_id, 'text_content', bio_id, 'biological_entity', 'текстовое описание')
                        )
                        print(f"✅ Создана связь между текстом {text_id} и биологической сущностью {bio_id}")
            
            print(f"Successfully processed text ID: {text_id}, in_stoplist: {in_stoplist_value}")
            return text_id
            
        except Exception as e:
            print(f"Error processing text: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_weather_for_image(self, feature_photo):
        """Извлекает и форматирует погодные условия из feature_photo"""
        weather_conditions = []
        feature_data = {}
        
        cloudiness = feature_photo.get('cloudiness')
        if cloudiness and cloudiness != 'Неопределено':
            weather_conditions.append(f"Облачность: {cloudiness}")
        
        temperature = feature_photo.get('temperature')
        if temperature and temperature != 'Неопределено':
            weather_conditions.append(f"Температура: {temperature}")
            try:
                temp_match = re.search(r'(\d+)', str(temperature))
                if temp_match:
                    feature_data['temperature_approx'] = float(temp_match.group(1))
            except:
                pass
        
        wind = feature_photo.get('wind')
        if wind and wind != 'Неопределено':
            weather_conditions.append(f"Ветер: {wind}")
            feature_data['windy'] = any(word in wind.lower() for word in ['ветер', 'ветрен', 'ветрено', 'wind'])
        
        precipitation = feature_photo.get('precipitation')
        if precipitation and precipitation != 'Неопределено':
            weather_conditions.append(f"Осадки: {precipitation}")
            feature_data['rain'] = any(word in precipitation.lower() for word in ['дождь', 'дожд', 'осадк', 'rain'])
        
        weather_text = ', '.join(weather_conditions) if weather_conditions else None
        
        return weather_text, feature_data

    def process_image(self, resource):
        """Обработка изображений с созданием географических сущностей и обработкой погодных условий"""
        try:
            identificator = resource['identificator']
            access = resource.get('access_options', {})
            feature_photo = resource.get('featurePhoto', {})
            name_info = identificator.get('name', {})
            
            title = self.get_title(resource)
            
            file_path = access.get('file_path')
            if not file_path:
                file_path = feature_photo.get('file_path')
            
            in_stoplist_value = self.safe_convert_in_stoplist(resource.get('in_stoplist'))
            
            baikal_relation = resource.get('baikal_relation')
            if isinstance(baikal_relation, str):
                baikal_relation = [baikal_relation]
            elif baikal_relation is None:
                baikal_relation = []
            
            image_feature_data = {}
            if feature_photo:
                image_feature_data.update({
                    'date': feature_photo.get('date'),
                    'season': feature_photo.get('season'),
                    'habitat': feature_photo.get('habitat'),
                    'location': feature_photo.get('location', {}),
                    'cloudiness': feature_photo.get('cloudiness'),
                    'fauna_type': feature_photo.get('fauna_type'),
                    'flora_type': feature_photo.get('flora_type'),
                    'name_photo': feature_photo.get('name_photo'),
                    'name_object': feature_photo.get('name_object'),
                    'author_photo': feature_photo.get('author_photo'),
                    'image_caption': feature_photo.get('image_caption'),
                    'classification_info': feature_photo.get('classification_info', {}),
                    'flower_and_fruit_info': feature_photo.get('flower_and_fruit_info', {}),
                    'yolo_detected_objects': feature_photo.get('yolo_detected_objects', [])
                })
            
            image_feature_data['in_stoplist'] = in_stoplist_value
            image_feature_data['baikal_relation'] = baikal_relation
            image_feature_data['information_type'] = resource.get('information_type', '')
            image_feature_data['information_subtype'] = resource.get('information_subtype', '')
            
            image_feature_data['identificator'] = {
                'id': identificator.get('id'),
                'uri': identificator.get('uri'),
                'name': {
                    'common': name_info.get('common'),
                    'source': name_info.get('source')
                }
            }
            image_feature_data['access_options'] = {
                'author': access.get('author'),
                'source_url': access.get('source_url'),
                'original_title': access.get('original_title')
            }
            
            existing_image_id = self.find_existing_image(file_path, title, image_feature_data)
            if existing_image_id:
                print(f"⚠️  Изображение уже существует в БД (id: {existing_image_id}), пропускаем...")
                return existing_image_id
            
            self.cur.execute(
                "INSERT INTO image_content (title, description, feature_data) "
                "VALUES (%s, %s, %s) RETURNING id",
                (
                    title,
                    image_feature_data.get('image_caption', ''),
                    Json(image_feature_data)
                )
            )
            image_id = self.cur.fetchone()[0]
            entity_type = 'image_content'
            
            self.add_reliability('image_content', image_id, name_info.get('source'))
            
            self.cur.execute(
                "INSERT INTO entity_identifier (url, file_path, name_ru, name_en, name_latin) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (
                    access.get('source_url', ''),
                    file_path,
                    title,
                    name_info.get('en_name'),
                    name_info.get('scientific')
                )
            )
            identifier_id = self.cur.fetchone()[0]
            
            self.cur.execute(
                "INSERT INTO entity_identifier_link (entity_id, entity_type, identifier_id) "
                "VALUES (%s, %s, %s)",
                (image_id, entity_type, identifier_id)
            )
            
            author_name = access.get('author')
            if not author_name:
                author_name = feature_photo.get('author_photo')
            
            print(f"🔍 Поиск автора: {author_name}")
            if author_name:
                author_id = self.get_or_create_author(author_name)
                print(f"✅ ID автора получен: {author_id}")
                if author_id:
                    self.cur.execute(
                        "INSERT INTO entity_author (entity_id, entity_type, author_id) "
                        "VALUES (%s, %s, %s) ON CONFLICT (entity_id, entity_type, author_id) DO NOTHING",
                        (image_id, entity_type, author_id)
                    )
                    print(f"✅ Связь с автором создана для image_id={image_id}")
            else:
                print("⚠️  Автор не найден в ресурсе")
            
            date_taken = image_feature_data.get('date')
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
                        "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (image_id, entity_type, temporal_id)
                    )
            
            if any(field in feature_photo for field in ['cloudiness', 'temperature', 'wind', 'precipitation']):
                weather_text, weather_features = self._process_weather_for_image(feature_photo)
                
                if weather_text:
                    try:
                        windy = weather_features.get('windy', False)
                        rain = weather_features.get('rain', False)
                        temperature_approx = weather_features.get('temperature_approx')
                        
                        self.cur.execute(
                            """
                            INSERT INTO weather_reference 
                            (weather_conditions, windy, rain, temperature_approx, timestamp)
                            VALUES (%s, %s, %s, %s, %s) 
                            RETURNING id
                            """,
                            (weather_text, windy, rain, temperature_approx, parsed_date or datetime.now())
                        )
                        weather_id = self.cur.fetchone()[0]
                        
                        self.cur.execute(
                            "INSERT INTO entity_weather (entity_id, entity_type, weather_id) "
                            "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                            (image_id, entity_type, weather_id)
                        )
                        
                        print(f"✅ Обработаны погодные условия для изображения {image_id}: {weather_text}")
                    except Exception as e:
                        print(f"⚠️ Ошибка обработки погодных условий: {e}")
            
            classification = image_feature_data.get('classification_info', {})
            result_info = classification.get('result', {}) if classification else {}
            
            information_subtype = resource.get('information_subtype')
            if not information_subtype:
                flora_type = image_feature_data.get('flora_type')
                fauna_type = image_feature_data.get('fauna_type')
                
                if flora_type and flora_type != 'Неопределено':
                    information_subtype = 'Объект флоры'
                elif fauna_type and fauna_type != 'Неопределено':
                    information_subtype = 'Объект фауны'
                elif 'flora' in str(resource.get('information_type', '')).lower():
                    information_subtype = 'Объект флоры'
                elif 'fauna' in str(resource.get('information_type', '')).lower():
                    information_subtype = 'Объект фауны'
            
            common_name = name_info.get('common') or result_info.get('species') or result_info.get('name')
            classification = image_feature_data.get('classification_info', {})
            result_info = classification.get('result', classification) if classification else {}

            has_classification = bool(result_info and any(
                v for k, v in result_info.items() 
                if k != 'source' and v and str(v).strip()
            ))
            if has_classification or (common_name and information_subtype):
                bio_id = self.process_biological_entity(
                    image_id, 
                    entity_type,
                    name_info,
                    result_info,
                    image_feature_data,
                    information_subtype
                )
                
                if bio_id:
                    print(f"✅ Создана/найдена биологическая сущность {bio_id} для изображения {image_id}")
            
            location = image_feature_data.get('location', {})
            if location:
                geo_id = self.process_geographical_data(
                    image_id, 
                    entity_type,
                    location,
                    name_info
                )
                if geo_id:
                    print(f"✅ Создана/найдена географическая сущность {geo_id} для изображения {image_id}")
            
            print(f"✅ Создано изображение: {title} (id: {image_id}), путь: {file_path}")
            return image_id
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def find_existing_image(self, file_path, title, feature_data):
        """Ищет существующее изображение по URL/пути к файлу и ключевым полям"""
        try:
            name_photo = feature_data.get('name_photo')
            author_photo = feature_data.get('author_photo')
            date = feature_data.get('date')
            
            if file_path:
                self.cur.execute(
                    "SELECT eil.entity_id FROM entity_identifier ei "
                    "JOIN entity_identifier_link eil ON ei.id = eil.identifier_id "
                    "WHERE ei.file_path = %s AND eil.entity_type = 'image_content' "
                    "LIMIT 1",
                    (file_path,)
                )
                result = self.cur.fetchone()
                if result:
                    print(f"🔍 Найдено существующее изображение по file_path: {file_path}")
                    return result[0]
            
            if name_photo and author_photo and date:
                self.cur.execute(
                    "SELECT id FROM image_content "
                    "WHERE feature_data->>'name_photo' = %s "
                    "AND feature_data->>'author_photo' = %s "
                    "AND feature_data->>'date' = %s "
                    "LIMIT 1",
                    (name_photo, author_photo, date)
                )
                result = self.cur.fetchone()
                if result:
                    print(f"🔍 Найдено существующее изображение по name_photo: {name_photo}")
                    return result[0]
            
            if title:
                self.cur.execute(
                    "SELECT id FROM image_content "
                    "WHERE title = %s "
                    "AND (feature_data->>'name_photo' = %s OR %s IS NULL) "
                    "LIMIT 1",
                    (title, name_photo, name_photo)
                )
                result = self.cur.fetchone()
                if result:
                    print(f"🔍 Найдено существующее изображение по title: {title}")
                    return result[0]
            
            source_url = feature_data.get('access_options', {}).get('source_url')
            if source_url:
                self.cur.execute(
                    "SELECT eil.entity_id FROM entity_identifier ei "
                    "JOIN entity_identifier_link eil ON ei.id = eil.identifier_id "
                    "WHERE ei.url = %s AND eil.entity_type = 'image_content' "
                    "LIMIT 1",
                    (source_url,)
                )
                result = self.cur.fetchone()
                if result:
                    print(f"🔍 Найдено существующее изображение по source_url: {source_url}")
                    return result[0]
            
            return None
            
        except Exception as e:
            print(f"Error finding existing image: {e}")
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
        """Получает название биологической сущности для карт с приоритетом animal_russian_name/plant_russian_name"""
        
        if resource.get('information_subtype') == "Объект фауны":
            animal_russian_name = resource.get('animal_russian_name')
            if animal_russian_name and animal_russian_name.strip():
                return animal_russian_name.strip()
        
        plant_russian_name = resource.get('plant_russian_name')
        if plant_russian_name and plant_russian_name.strip():
            return plant_russian_name.strip()
        
        common_name = resource['identificator'].get('name', {}).get('common', '')
        if common_name:
            cleaned_name = common_name.replace('Место обитания', '').replace('место обитания', '').strip()
            if cleaned_name:
                return cleaned_name
        
        resource_id = resource['identificator'].get('id', '')
        if resource_id.startswith('GEO_'):
            return resource_id.replace('GEO_', '').replace('_', ' ').strip()
        
        return 'Неизвестный вид'
    
    def determine_biological_type(self, feature_data):
        """Определяет тип биологического объекта на основе flora_type/fauna_type"""
        if not feature_data:
            return None
        
        fauna_type = feature_data.get('fauna_type')
        if fauna_type and fauna_type.strip():
            return "Объект фауны"
        
        flora_type = feature_data.get('flora_type')
        if flora_type and flora_type.strip():
            return "Объект флоры"
        
        information_subtype = feature_data.get('information_subtype')
        if information_subtype and information_subtype.strip():
            return information_subtype
        
        return None

    def process_map(self, resource):
        identificator = resource['identificator']
        name_info = identificator.get('name', {})
        geo_synonyms = resource.get('geo_synonyms', [])
        
        if resource.get('information_subtype') == "Объект фауны":
            common_name = resource.get('animal_russian_name') or self._get_biological_name_from_map(resource)
            scientific_name = resource.get('animal_latin_name')
        else:
            common_name = self._get_biological_name_from_map(resource)
            scientific_name = resource.get('plant_latin_name')
        
        information_subtype = resource.get('information_subtype')
        feature_data = resource.get('feature_data', {})
        
        bio_id = self._process_biological_entity(
            common_name,
            scientific_name,
            name_info.get('source'),
            resource.get('in_stoplist', False),
            information_subtype,
            feature_data
        )

        for geo_name in geo_synonyms:
            if not geo_name:
                continue
                
            simplified_name = self.simplify_geo_name(geo_name)
            
            geo_data = self.get_geo_data(simplified_name)
            
            if geo_data and 'geometry' in geo_data:
                full_geo_id = self.process_geo_mention(None, None, geo_name, name_info)
                
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
                
                self.cur.execute(
                    """
                    INSERT INTO entity_geo 
                    (entity_id, entity_type, geographical_entity_id)
                    VALUES (%s, 'map_content', %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (map_id, full_geo_id)
                )
                
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
        
        if common_name:
            common_name = self.normalize_species_name(common_name)
        
        bio_id = self.find_biological_entity(common_name, scientific_name)
        
        biological_type = information_subtype
        if not biological_type and feature_data:
            biological_type = self.determine_biological_type(feature_data)
        
        feature_data_dict = {}
        if in_stoplist_value is not None:
            feature_data_dict['in_stoplist'] = self.safe_convert_in_stoplist(in_stoplist_value)
        if feature_data:
            feature_data_dict.update({
                'flora_type': feature_data.get('flora_type'),
                'fauna_type': feature_data.get('fauna_type'),
                'information_subtype': information_subtype
            })
        
        if bio_id:
            if biological_type:
                self.cur.execute(
                    "UPDATE biological_entity SET type = %s WHERE id = %s AND type IS NULL",
                    (biological_type, bio_id)
                )
            return bio_id
            
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
        
        for name in filter(None, [common_name, scientific_name]):
            self.bio_entity_cache[name] = bio_id
            
        self.add_reliability('biological_entity', bio_id, source)
        
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

    def import_resources(self, json_file, incremental_mode=False):
        """Основной метод импорта с поддержкой инкрементального режима"""
        self.incremental_mode = incremental_mode
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        resources = data['resources']
        total_resources = len(resources)
        
        import sys
        sys.stdout.write(f"📊 Начинаем импорт {total_resources} ресурсов\n")
        sys.stdout.write(f"🔧 Режим: {'инкрементальный' if incremental_mode else 'полный'}\n")
        sys.stdout.flush()
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        existing_hashes = set()
        if self.incremental_mode:
            sys.stdout.write("🔍 Проверка существующих ресурсов...\n")
            sys.stdout.flush()
            existing_hashes = self.get_existing_resource_hashes()
            sys.stdout.write(f"📊 Найдено {len(existing_hashes)} существующих записей\n")
            sys.stdout.flush()
        
        start_time = time.time()
        
        for i, resource in enumerate(resources, 1):
            try:
                if i <= 5 or i % 5 == 0 or i == total_resources:
                    elapsed_time = time.time() - start_time
                    estimated_total = (elapsed_time / i) * total_resources if i > 0 else 0
                    remaining_time = estimated_total - elapsed_time
                    
                    sys.stdout.write(f"\n📊 Прогресс: {i}/{total_resources}\n")
                    sys.stdout.write(f"⏱️  Прошло: {elapsed_time:.1f}с, Осталось: ~{remaining_time:.1f}с\n")
                    self.log_progress(i, total_resources)
                
                rtype = resource['type']
                sys.stdout.write(f"\n📋 Обработка ресурса {i}/{total_resources}: {rtype}\n")
                sys.stdout.flush()
                
                if self.incremental_mode:
                    resource_hash = self.calculate_resource_hash(resource)
                    if resource_hash in existing_hashes:
                        sys.stdout.write(f"⏭️  Ресурс уже существует, пропускаем...\n")
                        sys.stdout.flush()
                        skipped_count += 1
                        continue
                
                if rtype == 'Изображение':
                    result = self.process_image(resource)
                elif rtype == 'Текст':
                    result = self.process_text(resource)
                elif rtype == 'Картографическая информация':
                    result = self.process_map(resource)
                elif rtype == 'Географический объект':
                    result = self.process_geographical_object(resource)
                else:
                    sys.stdout.write(f"⚠️  Неизвестный тип ресурса: {rtype}\n")
                    sys.stdout.flush()
                    result = None
                
                if result:
                    self.conn.commit()
                    success_count += 1
                    sys.stdout.write(f"✅ Успешно обработан (ID: {result})\n")
                    sys.stdout.flush()
                else:
                    self.conn.rollback()
                    error_count += 1
                    sys.stdout.write(f"❌ Ошибка обработки ресурса\n")
                    sys.stdout.flush()
                
            except Exception as e:
                sys.stdout.write(f"❌ Ошибка обработки ресурса {i}: {e}\n")
                sys.stdout.flush()
                import traceback
                traceback.print_exc()
                self.conn.rollback()
                error_count += 1
                self.entity_cache = {}
                self.author_cache = {}
                self.bio_entity_cache = {}

        total_time = time.time() - start_time
        
        sys.stdout.write(f"\n" + "="*60 + "\n")
        sys.stdout.write(f"🎉 ИМПОРТ ЗАВЕРШЕН\n")
        sys.stdout.write(f"="*60 + "\n")
        sys.stdout.write(f"✅ Успешно: {success_count}\n")
        sys.stdout.write(f"⏭️  Пропущено (дубликаты): {skipped_count}\n")
        sys.stdout.write(f"❌ Ошибок: {error_count}\n")
        sys.stdout.write(f"⏱️  Общее время: {total_time:.1f} секунд\n")
        sys.stdout.write(f"📊 Среднее время на ресурс: {total_time/total_resources if total_resources > 0 else 0:.2f} секунд\n")
        sys.stdout.write(f"🔧 Режим: {'инкрементальный' if incremental_mode else 'полный'}\n")
        sys.stdout.write(f"="*60 + "\n")
        sys.stdout.flush()
        
        return success_count, skipped_count, error_count
    
    def calculate_resource_hash(self, resource):
        """Создает хэш для ресурса для сравнения"""
        import hashlib
        import json
        
        hash_data = {
            'type': resource.get('type'),
            'identificator': resource.get('identificator', {}),
            'access_options': resource.get('access_options', {}),
            'feature_data': resource.get('feature_data', {}) if resource.get('type') != 'Изображение' else None,
            'featurePhoto': resource.get('featurePhoto', {}) if resource.get('type') == 'Изображение' else None
        }
        
        data_str = json.dumps(hash_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    def get_existing_resource_hashes(self):
        """Получает хэши уже существующих ресурсов из базы данных"""
        existing_hashes = set()
        
        try:
            self.cur.execute(
                """
                SELECT md5(concat(
                    COALESCE(feature_data->>'name_photo', ''),
                    COALESCE(feature_data->>'author_photo', ''),
                    COALESCE(feature_data->>'date', ''),
                    COALESCE(feature_data->>'image_caption', '')
                )) as hash
                FROM image_content
                """
            )
            for row in self.cur.fetchall():
                existing_hashes.add(row[0])
            
            self.cur.execute(
                """
                SELECT md5(concat(
                    COALESCE(title, ''),
                    COALESCE(feature_data->>'source', ''),
                    COALESCE(feature_data->>'baikal_relation', ''),
                    COALESCE(content, '')
                )) as hash
                FROM text_content
                """
            )
            for row in self.cur.fetchall():
                existing_hashes.add(row[0])
            
            self.cur.execute(
                """
                SELECT md5(concat(
                    COALESCE(name_ru, ''),
                    COALESCE(type, ''),
                    COALESCE(feature_data->>'source', ''),
                    COALESCE(feature_data->>'geo_entity_type', '')
                )) as hash
                FROM geographical_entity
                """
            )
            for row in self.cur.fetchall():
                existing_hashes.add(row[0])
                
        except Exception as e:
            print(f"Error fetching existing resource hashes: {e}")
        
        return existing_hashes
            
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

def main():
    """Основная функция с парсингом аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Импорт ресурсов в базу данных')
    parser.add_argument('--full', action='store_true',
                       help='Полная перезагрузка базы данных (очистка и создание заново)')
    parser.add_argument('--incremental', action='store_true',
                       help='Инкрементальный импорт без очистки базы')
    parser.add_argument('--json-file', default='../../json_files/resources_dist.json',
                       help='Путь к JSON файлу с ресурсами')
    
    args = parser.parse_args()
    
    if not args.full and not args.incremental:
        args.incremental = True
    
    if args.full and args.incremental:
        print("⚠️  Указаны оба режима --full и --incremental, используется --full")
        args.incremental = False
    
    importer = NewResourceImporter(incremental_mode=args.incremental)
    
    try:
        importer.connect()
        
        if args.full:
            print("🔄 Режим: полная перезагрузка базы данных")
            importer.cur.execute("""
                TRUNCATE TABLE 
                    image_content, text_content, map_content, biological_entity, 
                    geographical_entity, entity_relation, entity_author, author, 
                    entity_identifier, entity_identifier_link, reliability, entity_geo,
                    external_link, temporal_reference, entity_temporal, 
                    weather_reference, entity_weather, park_reference, entity_park,
                    ecological_reference, entity_ecological, territorial_reference,
                    entity_territorial
                CASCADE;
            """)
            importer.conn.commit()
            print("✅ Таблицы очищены")
        
        success, skipped, errors = importer.import_resources(args.json_file)
        
        importer.save_missing_geometry_objects()
        
        print(f"\nИмпорт завершен:")
        print(f"  Успешно: {success}")
        print(f"  Пропущено (дубликаты): {skipped}")
        print(f"  Ошибок: {errors}")
        print(f"  Режим: {'полный' if args.full else 'инкрементальный'}")
        
        return 0 if errors == 0 else 1
        
    except Exception as e:
        print(f"❌ Ошибка при импорте: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        importer.disconnect()

if __name__ == "__main__":
    sys.exit(main())