import json
import hashlib
import re
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from psycopg2.extras import Json as PgJson

from .interfaces import DatabaseClient, ResourceImporter


class EcoAssistantImporter(ResourceImporter):
    def __init__(self, client: DatabaseClient, geodb_path: Optional[Path] = None):
        self._client = client
        self._geodb_path = geodb_path or Path("/var/www/salut_bot/json_files/geodb.json")
        self._logger = logging.getLogger(__name__)
        self._error_logger = logging.getLogger('errors')
        self._geodb_data = self._load_geodb()
        self._species_synonyms = self._load_species_synonyms()
        self._object_cache = {}
        self._property_value_cache = {}
        self._modality_cache = {}
        self._bibliographic_cache = {}
        self._generation_cache = {}
        self._support_metadata_cache = {}
        self._missing_geometry_objects = set()
        self._processed_count = 0
        self._error_count = 0

    def _load_geodb(self) -> Dict:
        try:
            with open(self._geodb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._logger.info(f"Loaded geodb with {len(data)} entries")
                return data
        except FileNotFoundError:
            self._logger.warning(f"Geodb file not found: {self._geodb_path}")
            return {}
        except Exception as e:
            self._logger.error(f"Error loading geodb: {e}")
            return {}

    def _load_species_synonyms(self) -> Dict:
        synonyms_path = Path(__file__).parent.parent / "json_files" / "object_synonyms.json"
        try:
            with open(synonyms_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                result = data.get('biological_entity', {})
                self._logger.info(f"Loaded {len(result)} species synonyms")
                return result
        except FileNotFoundError:
            self._logger.warning(f"Synonyms file not found: {synonyms_path}")
            return {}
        except Exception as e:
            self._logger.error(f"Error loading synonyms: {e}")
            return {}

    def import_resources(self, json_file: str, incremental: bool = False) -> Dict[str, int]:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            self._logger.error(f"JSON file not found: {json_file}")
            return {'success': 0, 'skipped': 0, 'errors': 0}
        except json.JSONDecodeError as e:
            self._logger.error(f"Invalid JSON in {json_file}: {e}")
            return {'success': 0, 'skipped': 0, 'errors': 0}
            
        resources = data.get('resources', [])
        total = len(resources)
        success = 0
        skipped = 0
        errors = 0
        
        self._logger.info(f"Starting import of {total} resources, incremental={incremental}")

        for i, res in enumerate(resources, 1):
            try:
                resource_id = res.get('identificator', {}).get('id', 'unknown')
                resource_type = res.get('type', 'unknown')
                
                if i % 100 == 0:
                    self._logger.info(f"Progress: {i}/{total} ({i*100//total}%), errors: {errors}")
                
                if incremental:
                    r_hash = self._calculate_resource_hash(res)
                    if self._resource_exists(r_hash):
                        skipped += 1
                        if i % 1000 == 0:
                            self._logger.debug(f"Skipped duplicate: {resource_id}")
                        continue

                rtype = res.get('type')
                result = None
                
                try:
                    if rtype == 'Изображение':
                        result = self._process_image(res)
                    elif rtype == 'Текст':
                        result = self._process_text(res)
                    elif rtype == 'Картографическая информация':
                        result = self._process_map(res)
                    elif rtype == 'Географический объект':
                        result = self._process_geographical_object(res)
                    else:
                        self._logger.warning(f"Unknown resource type: {rtype} for {resource_id}")
                        result = None
                        
                except Exception as e:
                    error_msg = f"Error processing {resource_type} resource {resource_id}: {e}\n{traceback.format_exc()}"
                    self._error_logger.error(error_msg)
                    self._logger.debug(f"Failed: {resource_id} - {str(e)[:100]}")
                    raise

                if result:
                    success += 1
                    if incremental:
                        row = self._client.fetchone(
                        "SELECT support_metadata_id FROM eco_assistant.resource WHERE id = %s",
                        (result,)
                    )
                        if row:
                            self._store_resource_hash(row[0], r_hash)
                    self._logger.debug(f"Successfully imported: {resource_id} -> {result}")
                else:
                    errors += 1
                    self._logger.warning(f"Resource {resource_id} returned no result")
                    
            except Exception as e:
                errors += 1
                self._logger.error(f"Critical error processing resource {i}: {e}")
                self._client.rollback()
                continue

        self._save_missing_geometry_objects()
        self._logger.info(f"Import finished. Success: {success}, Skipped: {skipped}, Errors: {errors}")
        return {'success': success, 'skipped': skipped, 'errors': errors}

    def _normalize_species_name(self, name: str) -> str:
        if not name:
            return name
        name_lower = name.strip().lower()
        for main_name, synonyms in self._species_synonyms.items():
            if name_lower == main_name.lower():
                return main_name
            for synonym in synonyms:
                if name_lower == synonym.lower():
                    return main_name
        return name

    def _get_or_create_object(self, name: str, object_type: str, classification_identifier: Optional[str] = None) -> int:
        cache_key = (name, object_type)
        if cache_key in self._object_cache:
            return self._object_cache[cache_key]

        row = self._client.fetchone(
            "SELECT o.id FROM eco_assistant.object o "
            "JOIN eco_assistant.object_description od ON o.id = od.object_id "
            "WHERE o.name = %s AND od.object_type = %s",
            (name, object_type)
        )
        if row:
            obj_id = row[0]
            self._object_cache[cache_key] = obj_id
            return obj_id

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.object (name) VALUES (%s) RETURNING id",
            (name,)
        )
        obj_id = row[0]
        self._client.fetchone(
            "INSERT INTO eco_assistant.object_description (object_id, classification_identifier, object_type) "
            "VALUES (%s, %s, %s) RETURNING id",
            (obj_id, classification_identifier, object_type)
        )
        self._client.commit()
        self._object_cache[cache_key] = obj_id
        return obj_id

    def _get_or_create_property_value(self, value: str) -> int:
        if value in self._property_value_cache:
            return self._property_value_cache[value]

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.property_value (value) VALUES (%s) "
            "ON CONFLICT (value_md5) DO NOTHING RETURNING id",
            (value,)
        )
        if row:
            pv_id = row[0]
            self._property_value_cache[value] = pv_id
            return pv_id

        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.property_value WHERE value_md5 = md5(%s)",
            (value,)
        )
        pv_id = row[0]
        self._property_value_cache[value] = pv_id
        return pv_id
    
    def _add_object_property(self, object_id: int, object_type: str, property_name: str, value: str) -> None:
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.object_description WHERE object_id = %s AND object_type = %s",
            (object_id, object_type)
        )
        if not row:
            return
        desc_id = row[0]
        pv_id = self._get_or_create_property_value(value)
        self._client.execute(
            "INSERT INTO eco_assistant.object_property (object_description_id, property_name, object_type, property_value_id) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (desc_id, property_name, object_type, pv_id)
        )
        self._client.commit()

    def _add_object_synonym(self, object_id: int, object_type: str, synonym: str, language: str = 'ru') -> None:
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.object_description WHERE object_id = %s AND object_type = %s",
            (object_id, object_type)
        )
        if not row:
            return
        desc_id = row[0]
        self._client.execute(
            "INSERT INTO eco_assistant.object_synonym (object_description_id, synonym, language) "
            "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (desc_id, synonym, language)
        )
        self._client.commit()

    def _get_or_create_modality(self, modality_type: str) -> int:
        if modality_type in self._modality_cache:
            return self._modality_cache[modality_type]

        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.modality WHERE modality_type = %s",
            (modality_type,)
        )
        if row:
            mod_id = row[0]
            self._modality_cache[modality_type] = mod_id
            return mod_id

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.modality (modality_type) VALUES (%s) RETURNING id",
            (modality_type,)
        )
        mod_id = row[0]
        self._client.commit()
        self._modality_cache[modality_type] = mod_id
        return mod_id

    def _get_or_create_bibliographic(self, author: Optional[str], date: Optional[str], source: Optional[str],
                                     rights: Optional[str], reliability: Optional[str]) -> int:
        key = (author or '', date or '', source or '', rights or '', reliability or '')
        if key in self._bibliographic_cache:
            return self._bibliographic_cache[key]

        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.bibliographic WHERE "
            "COALESCE(author, '') = COALESCE(%s, '') AND "
            "COALESCE(date::text, '') = COALESCE(%s, '') AND "
            "COALESCE(source, '') = COALESCE(%s, '') AND "
            "COALESCE(rights, '') = COALESCE(%s, '') AND "
            "COALESCE(reliability, '') = COALESCE(%s, '')",
            (author, date, source, rights, reliability)
        )
        if row:
            bib_id = row[0]
            self._bibliographic_cache[key] = bib_id
            return bib_id

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.bibliographic (author, date, source, rights, reliability) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (author, date if date else None, source, rights, reliability)
        )
        bib_id = row[0]
        self._client.commit()
        self._bibliographic_cache[key] = bib_id
        return bib_id

    def _get_or_create_generation(self, generation_type: Optional[str], generation_tool: Optional[str],
                                  generation_params: Optional[Dict]) -> int:
        key = (generation_type or '', generation_tool or '', json.dumps(generation_params or {}, sort_keys=True))
        if key in self._generation_cache:
            return self._generation_cache[key]

        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.generation WHERE "
            "COALESCE(generation_type,'') = %s AND COALESCE(generation_tool,'') = %s "
            "AND COALESCE(generation_params::text,'') = %s",
            (generation_type, generation_tool, json.dumps(generation_params or {}))
        )
        if row:
            gen_id = row[0]
            self._generation_cache[key] = gen_id
            return gen_id

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.generation (generation_type, generation_tool, generation_params) "
            "VALUES (%s, %s, %s) RETURNING id",
            (generation_type, generation_tool, PgJson(generation_params) if generation_params else None)
        )
        gen_id = row[0]
        self._client.commit()
        self._generation_cache[key] = gen_id
        return gen_id

    def _get_or_create_support_metadata(self, parameters: Dict) -> int:
        key = json.dumps(parameters, sort_keys=True)
        if key in self._support_metadata_cache:
            return self._support_metadata_cache[key]

        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.support_metadata WHERE parameters::text = %s",
            (json.dumps(parameters),)
        )
        if row:
            meta_id = row[0]
            self._support_metadata_cache[key] = meta_id
            return meta_id

        row = self._client.fetchone(
            "INSERT INTO eco_assistant.support_metadata (parameters) VALUES (%s) RETURNING id",
            (PgJson(parameters),)
        )
        meta_id = row[0]
        self._client.commit()
        self._support_metadata_cache[key] = meta_id
        return meta_id

    def _create_resource(self, modality_id: int, bibliographic_id: int, generation_id: int,
                         support_metadata_id: int) -> int:
        row = self._client.fetchone(
            "INSERT INTO eco_assistant.resource (modality_id, bibliographic_id, generation_id, support_metadata_id) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (modality_id, bibliographic_id, generation_id, support_metadata_id)
        )
        resource_id = row[0]
        self._client.commit()
        return resource_id

    def _link_resource_to_object(self, resource_id: int, object_id: int, object_type: str) -> None:
        row = self._client.fetchone(
            "SELECT id FROM eco_assistant.object_description WHERE object_id = %s AND object_type = %s",
            (object_id, object_type)
        )
        if row:
            desc_id = row[0]
            self._client.execute(
                "INSERT INTO eco_assistant.resource_object (resource_id, object_description_id) "
                "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (resource_id, desc_id)
            )
            self._client.commit()

    def _resource_exists(self, resource_hash: str) -> bool:
        row = self._client.fetchone(
            "SELECT 1 FROM eco_assistant.support_metadata WHERE parameters->>'resource_hash' = %s",
            (resource_hash,)
        )
        return row is not None

    def _store_resource_hash(self, support_metadata_id: int, resource_hash: str) -> None:
        self._client.execute(
            "UPDATE eco_assistant.support_metadata SET parameters = parameters || %s WHERE id = %s",
            (PgJson({'resource_hash': resource_hash}), support_metadata_id)
        )
        self._client.commit()

    def _create_modality_text(self, modality_id: int, content: Dict) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.modality_text (modality_id, content) VALUES (%s, %s)",
            (modality_id, PgJson(content))
        )
        self._client.commit()

    def _create_modality_image(self, modality_id: int, url: Optional[str], file_path: Optional[str],
                               quality: Optional[str] = None, width: Optional[int] = None,
                               height: Optional[int] = None, format: Optional[str] = None) -> None:
        self._client.execute(
            "INSERT INTO eco_assistant.modality_image (modality_id, url, file_path, quality, width, height, format) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (modality_id, url, file_path, quality, width, height, format)
        )
        self._client.commit()

    def _create_modality_geodata(self, modality_id: int, geometry: Dict) -> None:
        geom_json = json.dumps(geometry)
        self._client.execute(
            "INSERT INTO eco_assistant.modality_geodata (modality_id, geometry) "
            "VALUES (%s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))",
            (modality_id, geom_json)
        )
        self._client.commit()

    def _calculate_resource_hash(self, resource: Dict) -> str:
        data = {
            'type': resource.get('type'),
            'identificator': resource.get('identificator'),
            'access_options': resource.get('access_options'),
            'feature_data': resource.get('feature_data') if resource.get('type') != 'Изображение' else None,
            'featurePhoto': resource.get('featurePhoto') if resource.get('type') == 'Изображение' else None,
        }
        return hashlib.md5(json.dumps(data, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()

    def _get_title(self, resource: Dict) -> str:
        common = resource['identificator'].get('name', {}).get('common')
        if common:
            return common
        original = resource.get('access_options', {}).get('original_title')
        if original:
            return original
        return resource['identificator'].get('id', 'Без названия')

    def _get_reliability_value(self, source: Optional[str]) -> str:
        if not source:
            return "общедоступная"
        src_low = source.lower()
        if "национальный парк" in src_low or "заповедник" in src_low:
            return "профильная организация"
        if "ai generation" in src_low or "википедия" in src_low:
            return "общедоступная"
        return "профильная организация"

    def _parse_date(self, date_str: Optional[str]) -> Optional[str]:
        if not date_str:
            return None
        try:
            date_str = re.sub(r'[·•]', ' ', date_str).strip()
            formats = [
                '%d.%m.%Y %H:%M', '%d.%m.%Y', '%d.%m.%y %H:%M', '%d.%m.%y',
                '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M', '%d/%m/%Y',
                '%d %m %Y %H:%M', '%d %m %Y'
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def _get_geo_data(self, name: str) -> Optional[Dict]:
        if name in self._geodb_data:
            return self._geodb_data[name]
        for n, d in self._geodb_data.items():
            if n.lower() == name.lower():
                return d
        if ',' in name:
            parts = [p.strip() for p in name.split(',')]
            for part in reversed(parts):
                if part in self._geodb_data:
                    return self._geodb_data[part]
                for n, d in self._geodb_data.items():
                    if n.lower() == part.lower():
                        return d
        return None

    def _simplify_geo_name(self, name: str) -> str:
        return name.split(',')[-1].strip() if ',' in name else name.strip()

    def _clean_coordinate(self, coord: Any) -> Optional[float]:
        if coord is None:
            return None
        if isinstance(coord, (int, float)):
            return float(coord)
        if isinstance(coord, str):
            try:
                return float(coord.strip())
            except ValueError:
                return None
        try:
            return float(str(coord))
        except (ValueError, TypeError):
            return None

    def _process_geographical_object(self, resource: Dict) -> Optional[int]:
        identificator = resource.get('identificator', {})
        name_info = identificator.get('name', {})
        common_name = name_info.get('common')
        if not common_name:
            return None

        object_type = resource.get('geo_entity_type', 'Географический объект')
        description = resource.get('description', '')
        coordinates = resource.get('coordinates', {})
        geo_synonyms = resource.get('geo_synonyms', [])
        in_stoplist = resource.get('in_stoplist', False)

        object_id = self._get_or_create_object(common_name, object_type, None)

        if description:
            self._add_object_property(object_id, object_type, 'описание', description)

        if coordinates:
            lat = self._clean_coordinate(coordinates.get('latitude'))
            lon = self._clean_coordinate(coordinates.get('longitude'))
            if lat is not None and lon is not None:
                self._add_object_property(object_id, object_type, 'координаты', f"{lat}, {lon}")

        for synonym in geo_synonyms:
            if synonym and synonym != common_name:
                self._add_object_synonym(object_id, object_type, synonym)

        author = resource.get('access_options', {}).get('author')
        source = name_info.get('source')
        reliability = self._get_reliability_value(source)
        bibliographic_id = self._get_or_create_bibliographic(author, None, source, None, reliability)

        generation_id = self._get_or_create_generation(None, None, None)

        support_params = {
            'in_stoplist': in_stoplist,
            'original_data': {
                'identificator': identificator,
                'feature_data': resource.get('feature_data', {}),
                'meta_info': resource.get('meta_info', {})
            }
        }
        support_metadata_id = self._get_or_create_support_metadata(support_params)

        modality_id = self._get_or_create_modality('Геоданные')

        geo_data = self._get_geo_data(common_name)
        if geo_data and 'geometry' in geo_data:
            self._create_modality_geodata(modality_id, geo_data['geometry'])
        else:
            lat = self._clean_coordinate(coordinates.get('latitude'))
            lon = self._clean_coordinate(coordinates.get('longitude'))
            if lat is not None and lon is not None:
                point_geom = {"type": "Point", "coordinates": [lon, lat]}
                self._create_modality_geodata(modality_id, point_geom)
            else:
                self._missing_geometry_objects.add(common_name)

        resource_id = self._create_resource(modality_id, bibliographic_id, generation_id, support_metadata_id)
        self._link_resource_to_object(resource_id, object_id, object_type)

        return resource_id

    def _get_biological_name_from_map(self, resource: Dict) -> str:
        if resource.get('information_subtype') == "Объект фауны":
            animal = resource.get('animal_russian_name')
            if animal:
                return animal
        plant = resource.get('plant_russian_name')
        if plant:
            return plant
        common = resource['identificator'].get('name', {}).get('common', '')
        if common:
            return common.replace('Место обитания', '').strip()
        return 'Неизвестный вид'

    def _process_map(self, resource: Dict) -> Optional[int]:
        identificator = resource.get('identificator', {})
        name_info = identificator.get('name', {})
        geo_synonyms = resource.get('geo_synonyms', [])
        common_name = self._get_biological_name_from_map(resource)
        scientific_name = resource.get('plant_latin_name') or resource.get('animal_latin_name')
        information_subtype = resource.get('information_subtype')
        in_stoplist = resource.get('in_stoplist', False)

        if common_name:
            common_name = self._normalize_species_name(common_name)

        object_type = information_subtype or 'Биологический объект'
        object_id = self._get_or_create_object(common_name, object_type, None)

        if scientific_name:
            self._add_object_synonym(object_id, object_type, scientific_name, 'la')

        for synonym in geo_synonyms:
            if synonym:
                self._add_object_synonym(object_id, object_type, synonym)

        author = resource.get('access_options', {}).get('author')
        source = name_info.get('source')
        reliability = self._get_reliability_value(source)
        bibliographic_id = self._get_or_create_bibliographic(author, None, source, None, reliability)

        generation_id = self._get_or_create_generation(None, None, None)

        support_params = {
            'in_stoplist': in_stoplist,
            'information_subtype': information_subtype,
            'original_data': {
                'identificator': identificator,
                'feature_data': resource.get('feature_data', {})
            }
        }
        support_metadata_id = self._get_or_create_support_metadata(support_params)

        modality_id = self._get_or_create_modality('Геоданные')

        for geo_name in geo_synonyms:
            if not geo_name:
                continue
            simplified = self._simplify_geo_name(geo_name)
            geo_data = self._get_geo_data(simplified)
            if geo_data and 'geometry' in geo_data:
                self._create_modality_geodata(modality_id, geo_data['geometry'])
                break

        resource_id = self._create_resource(modality_id, bibliographic_id, generation_id, support_metadata_id)
        self._link_resource_to_object(resource_id, object_id, object_type)

        return resource_id

    def _process_text(self, resource: Dict) -> Optional[int]:
        identificator = resource.get('identificator', {})
        name_info = identificator.get('name', {})
        title = self._get_title(resource)
        content = resource.get('content', '')
        brief_annotation = resource.get('brief_annotation', '')
        structured_data = resource.get('structured_data', {})
        information_type = resource.get('information_type')
        in_stoplist = resource.get('in_stoplist', False)
        geo_synonyms = resource.get('geo_synonyms', [])

        object_type = information_type or 'Текстовый ресурс'
        object_id = self._get_or_create_object(title, object_type, None)

        if brief_annotation:
            self._add_object_property(object_id, object_type, 'аннотация', brief_annotation)

        for synonym in geo_synonyms:
            if synonym:
                self._add_object_synonym(object_id, object_type, synonym)

        author = resource.get('access_options', {}).get('author')
        source = name_info.get('source')
        reliability = self._get_reliability_value(source)
        bibliographic_id = self._get_or_create_bibliographic(author, None, source, None, reliability)

        generation_id = self._get_or_create_generation(None, None, None)

        support_params = {
            'in_stoplist': in_stoplist,
            'information_type': information_type,
            'original_data': {
                'identificator': identificator,
                'access_options': resource.get('access_options', {})
            }
        }
        support_metadata_id = self._get_or_create_support_metadata(support_params)

        modality_id = self._get_or_create_modality('Текст')

        text_content = {
            'content': content,
            'structured_data': structured_data,
            'title': title,
            'brief_annotation': brief_annotation
        }
        self._create_modality_text(modality_id, text_content)

        resource_id = self._create_resource(modality_id, bibliographic_id, generation_id, support_metadata_id)
        self._link_resource_to_object(resource_id, object_id, object_type)

        return resource_id

    def _process_weather_for_image(self, feature_photo: Dict) -> Tuple[Optional[str], Dict]:
        weather_conditions = []
        weather_data = {}

        cloudiness = feature_photo.get('cloudiness')
        if cloudiness and cloudiness != 'Неопределено':
            weather_conditions.append(f"Облачность: {cloudiness}")

        temperature = feature_photo.get('temperature')
        if temperature and temperature != 'Неопределено':
            weather_conditions.append(f"Температура: {temperature}")
            temp_match = re.search(r'(\d+)', str(temperature))
            if temp_match:
                weather_data['temperature_approx'] = float(temp_match.group(1))

        wind = feature_photo.get('wind')
        if wind and wind != 'Неопределено':
            weather_conditions.append(f"Ветер: {wind}")
            weather_data['windy'] = any(word in wind.lower() for word in ['ветер', 'ветрен', 'ветрено'])

        precipitation = feature_photo.get('precipitation')
        if precipitation and precipitation != 'Неопределено':
            weather_conditions.append(f"Осадки: {precipitation}")
            weather_data['rain'] = any(word in precipitation.lower() for word in ['дождь', 'дожд', 'осадк'])

        weather_text = ', '.join(weather_conditions) if weather_conditions else None
        return weather_text, weather_data

    def _process_image(self, resource: Dict) -> Optional[int]:
        identificator = resource.get('identificator', {})
        name_info = identificator.get('name', {})
        access_options = resource.get('access_options', {})
        feature_photo = resource.get('featurePhoto', {})
        title = self._get_title(resource)
        information_type = resource.get('information_type', 'Изображение')
        information_subtype = resource.get('information_subtype')
        in_stoplist = resource.get('in_stoplist', False)

        common_name = name_info.get('common')
        if common_name:
            common_name = self._normalize_species_name(common_name)

        object_type = information_subtype or information_type
        object_id = None
        if common_name:
            object_id = self._get_or_create_object(common_name, object_type, None)

        author = access_options.get('author') or feature_photo.get('author_photo')
        source = name_info.get('source')
        reliability = self._get_reliability_value(source)
        date_taken = feature_photo.get('date')
        parsed_date = self._parse_date(date_taken)

        bibliographic_id = self._get_or_create_bibliographic(author, parsed_date, source, None, reliability)

        generation_id = self._get_or_create_generation(None, None, None)

        support_params = {
            'in_stoplist': in_stoplist,
            'information_type': information_type,
            'information_subtype': information_subtype,
            'original_data': {
                'identificator': identificator,
                'access_options': access_options,
                'feature_photo': feature_photo
            }
        }
        support_metadata_id = self._get_or_create_support_metadata(support_params)

        modality_id = self._get_or_create_modality('Изображение')

        file_path = access_options.get('file_path') or feature_photo.get('file_path')
        url = access_options.get('source_url')
        self._create_modality_image(modality_id, url, file_path)

        resource_id = self._create_resource(modality_id, bibliographic_id, generation_id, support_metadata_id)

        if object_id and common_name:
            self._link_resource_to_object(resource_id, object_id, object_type)

        classification_info = feature_photo.get('classification_info', {})
        if classification_info:
            result_info = classification_info.get('result', {})
            for key, value in result_info.items():
                if value and key not in ['source']:
                    self._add_object_property(object_id, object_type, key, str(value))

        weather_text, weather_data = self._process_weather_for_image(feature_photo)
        if weather_text:
            self._add_object_property(object_id, object_type, 'погодные условия', weather_text)

        location = feature_photo.get('location', {})
        if location:
            lat = self._clean_coordinate(location.get('latitude'))
            lon = self._clean_coordinate(location.get('longitude'))
            if lat is not None and lon is not None:
                self._add_object_property(object_id, object_type, 'координаты', f"{lat}, {lon}")

        return resource_id

    def _save_missing_geometry_objects(self, output_file: str = "missing_geometry_objects.json") -> None:
        if self._missing_geometry_objects:
            missing_list = list(self._missing_geometry_objects)
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(missing_list, f, ensure_ascii=False, indent=2)
            except Exception:
                pass