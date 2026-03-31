from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import json
import logging

from ..domain.entities import (
    Resource,
    ResourceImportResult,
    SupportMetadata,
    DbId,
    Object,
    ObjectNameSynonym,
    BibliographicData,
    CreationData,
    ResourceStatic,
    TextValue,
    ImageValue,
    GeodataValue,
    ResourceType,
    ObjectType,
)
from .interfaces import (
    ResourceRepository,
    ObjectRepository,
    ObjectTypeRepository,
    SynonymRepository,
    ModalityRepository,
    BibliographicRepository,
    CreationRepository,
    ResourceStaticRepository,
    SupportMetadataRepository,
    SpeciesNameNormalizer,
)


@dataclass
class ImportResourceUseCase:
    resource_repo: ResourceRepository
    object_repo: ObjectRepository
    object_type_repo: ObjectTypeRepository
    synonym_repo: SynonymRepository
    modality_repo: ModalityRepository
    bibliographic_repo: BibliographicRepository
    creation_repo: CreationRepository
    resource_static_repo: ResourceStaticRepository
    metadata_repo: SupportMetadataRepository
    species_normalizer: SpeciesNameNormalizer

    _logger = logging.getLogger(__name__)

    def execute(self, resource_data: Dict[str, Any], incremental: bool = False) -> Optional[int]:
        if incremental:
            resource_hash = self._calculate_hash(resource_data)
            if self.resource_repo.resource_exists_by_hash(resource_hash):
                return None
        else:
            resource_hash = None

        resource_type = resource_data.get('type')

        if resource_type == ResourceType.IMAGE.value:
            return self._import_image(resource_data, resource_hash)
        elif resource_type == ResourceType.TEXT.value:
            return self._import_text(resource_data, resource_hash)
        elif resource_type == ResourceType.MAP.value:
            return self._import_map(resource_data, resource_hash)
        elif resource_type == ResourceType.GEOGRAPHICAL_OBJECT.value:
            return self._import_geographical_object(resource_data, resource_hash)
        else:
            self._logger.warning(f"Unknown resource type: {resource_type}")
            return None

    def _calculate_hash(self, resource: Dict[str, Any]) -> str:
        data = {
            'type': resource.get('type'),
            'identificator': resource.get('identificator'),
            'access_options': resource.get('access_options'),
            'feature_data': resource.get('feature_data') if resource.get('type') != 'Изображение' else None,
            'featurePhoto': resource.get('featurePhoto') if resource.get('type') == 'Изображение' else None,
        }
        return hashlib.md5(
            json.dumps(data, sort_keys=True, ensure_ascii=False).encode('utf-8')
        ).hexdigest()

    def _get_title(self, resource: Dict[str, Any]) -> str:
        common = resource['identificator'].get('name', {}).get('common')
        if common:
            return common
        original = resource.get('access_options', {}).get('original_title')
        if original:
            return original
        return resource['identificator'].get('id', 'Без названия')

    def _get_reliability(self, source: Optional[str]) -> str:
        if not source:
            return "общедоступная"

        src_low = source.lower()
        if "национальный парк" in src_low or "заповедник" in src_low:
            return "профильная организация"
        if "ai generation" in src_low or "википедия" in src_low:
            return "общедоступная"
        return "профильная организация"

    def _create_bibliographic(self, resource: Dict[str, Any], date: Optional[str] = None) -> int:
        access_options = resource.get('access_options', {})
        name_info = resource.get('identificator', {}).get('name', {})

        author = access_options.get('author')
        author_id = self.bibliographic_repo.get_or_create_author(author) if author else None

        source = name_info.get('source')
        source_id = self.bibliographic_repo.get_or_create_source(source) if source else None

        reliability = self._get_reliability(source)
        reliability_id = self.bibliographic_repo.get_or_create_reliability_level(reliability)

        bibliographic = BibliographicData(
            author_id=author_id,
            date=date,
            source_id=source_id,
            reliability_level_id=reliability_id
        )
        return self.bibliographic_repo.get_or_create(bibliographic)

    def _create_support_metadata(self, resource: Dict[str, Any],
                                  resource_hash: Optional[str] = None) -> int:
        metadata = SupportMetadata.from_resource(resource, resource_hash)
        return self.metadata_repo.get_or_create(metadata)

    def _get_or_create_object_type(self, object_type_name: str) -> ObjectType:
        return self.object_type_repo.get_or_create(object_type_name)

    def _get_or_create_object(self, name: str, object_type_id: int) -> Object:
        db_id = DbId.from_name_and_type(name, str(object_type_id))

        existing = self.object_repo.find_by_db_id(str(db_id), object_type_id)
        if existing:
            return existing

        object_obj = Object(
            db_id=db_id,
            object_type_id=object_type_id,
            object_properties={}
        )
        object_obj = self.object_repo.save(object_obj)

        synonym = self.synonym_repo.get_or_create(name, 'ru', True)
        self.object_repo.add_synonym_link(object_obj.id, synonym.id)

        return object_obj

    def _add_object_property(self, object_id: int, property_name: str, value: str) -> None:
        obj = self.object_repo.find_by_db_id(str(object_id), 0)
        if obj:
            obj.object_properties[property_name] = value
            self.object_repo.save(obj)

    def _import_image(self, resource: Dict[str, Any], resource_hash: Optional[str] = None) -> Optional[int]:
        bibliographic_id = self._create_bibliographic(resource)
        creation_id = self.creation_repo.get_or_create(CreationData())
        resource_static_id = self.resource_static_repo.get_or_create(
            ResourceStatic(bibliographic_id=bibliographic_id, creation_id=creation_id)
        )
        metadata_id = self._create_support_metadata(resource, resource_hash)

        modality = self.modality_repo.get_or_create_modality('Изображение', 'image_value')
        access_options = resource.get('access_options', {})
        feature_photo = resource.get('featurePhoto', {})

        image_value = ImageValue(
            url=access_options.get('source_url'),
            file_path=access_options.get('file_path') or feature_photo.get('file_path')
        )
        value_id = self.modality_repo.save_image_value(image_value)

        resource_obj = Resource(
            resource_static_id=resource_static_id,
            support_metadata_id=metadata_id
        )
        resource_id = self.resource_repo.save_resource(resource_obj)

        self.modality_repo.link_resource_value(resource_id, modality.id, value_id)

        name_info = resource.get('identificator', {}).get('name', {})
        common_name = name_info.get('common')

        if common_name:
            normalized_name = self.species_normalizer.normalize(common_name)
            information_subtype = resource.get('information_subtype')
            object_type_name = information_subtype or resource.get('information_type', 'Изображение')
            object_type = self._get_or_create_object_type(object_type_name)
            object_obj = self._get_or_create_object(normalized_name, object_type.id)
            self.resource_repo.link_resource_to_object(resource_id, object_obj.id)

            classification_info = feature_photo.get('classification_info', {})
            if classification_info:
                result_info = classification_info.get('result', {})
                for key, value in result_info.items():
                    if value and key not in ['source']:
                        self._add_object_property(object_obj.id, key, str(value))

            weather_text, _ = self._process_weather(feature_photo)
            if weather_text:
                self._add_object_property(object_obj.id, 'погодные условия', weather_text)

            location = feature_photo.get('location', {})
            if location:
                lat = self._clean_coordinate(location.get('latitude'))
                lon = self._clean_coordinate(location.get('longitude'))
                if lat is not None and lon is not None:
                    self._add_object_property(object_obj.id, 'координаты', f"{lat}, {lon}")

        return resource_id

    def _import_text(self, resource: Dict[str, Any], resource_hash: Optional[str] = None) -> Optional[int]:
        bibliographic_id = self._create_bibliographic(resource)
        creation_id = self.creation_repo.get_or_create(CreationData())
        resource_static_id = self.resource_static_repo.get_or_create(
            ResourceStatic(bibliographic_id=bibliographic_id, creation_id=creation_id)
        )
        metadata_id = self._create_support_metadata(resource, resource_hash)

        modality = self.modality_repo.get_or_create_modality('Текст', 'text_value')
        title = self._get_title(resource)
        text_content = {
            'content': resource.get('content', ''),
            'structured_data': resource.get('structured_data', {}),
            'title': title,
            'brief_annotation': resource.get('brief_annotation', '')
        }

        text_value = TextValue(content=text_content)
        value_id = self.modality_repo.save_text_value(text_value)

        resource_obj = Resource(
            resource_static_id=resource_static_id,
            support_metadata_id=metadata_id
        )
        resource_id = self.resource_repo.save_resource(resource_obj)

        self.modality_repo.link_resource_value(resource_id, modality.id, value_id)

        object_type_name = resource.get('information_type', 'Текстовый ресурс')
        object_type = self._get_or_create_object_type(object_type_name)
        object_obj = self._get_or_create_object(title, object_type.id)
        self.resource_repo.link_resource_to_object(resource_id, object_obj.id)

        brief_annotation = resource.get('brief_annotation')
        if brief_annotation:
            self._add_object_property(object_obj.id, 'аннотация', brief_annotation)

        return resource_id

    def _import_map(self, resource: Dict[str, Any], resource_hash: Optional[str] = None) -> Optional[int]:
        bibliographic_id = self._create_bibliographic(resource)
        creation_id = self.creation_repo.get_or_create(CreationData())
        resource_static_id = self.resource_static_repo.get_or_create(
            ResourceStatic(bibliographic_id=bibliographic_id, creation_id=creation_id)
        )
        metadata_id = self._create_support_metadata(resource, resource_hash)

        modality = self.modality_repo.get_or_create_modality('Геоданные', 'geodata_value')

        common_name = self._get_biological_name_from_map(resource)
        if common_name:
            common_name = self.species_normalizer.normalize(common_name)

        scientific_name = resource.get('plant_latin_name') or resource.get('animal_latin_name')
        information_subtype = resource.get('information_subtype')
        object_type_name = information_subtype or 'Биологический объект'
        object_type = self._get_or_create_object_type(object_type_name)

        object_obj = self._get_or_create_object(common_name, object_type.id)

        if scientific_name:
            synonym = self.synonym_repo.get_or_create(scientific_name, 'la', False)
            self.object_repo.add_synonym_link(object_obj.id, synonym.id)

        resource_obj = Resource(
            resource_static_id=resource_static_id,
            support_metadata_id=metadata_id
        )
        resource_id = self.resource_repo.save_resource(resource_obj)

        self.modality_repo.link_resource_value(resource_id, modality.id, None)
        self.resource_repo.link_resource_to_object(resource_id, object_obj.id)

        return resource_id

    def _import_geographical_object(self, resource: Dict[str, Any],
                                     resource_hash: Optional[str] = None) -> Optional[int]:
        identificator = resource.get('identificator', {})
        name_info = identificator.get('name', {})
        common_name = name_info.get('common')

        if not common_name:
            return None

        bibliographic_id = self._create_bibliographic(resource)
        creation_id = self.creation_repo.get_or_create(CreationData())
        resource_static_id = self.resource_static_repo.get_or_create(
            ResourceStatic(bibliographic_id=bibliographic_id, creation_id=creation_id)
        )
        metadata_id = self._create_support_metadata(resource, resource_hash)

        modality = self.modality_repo.get_or_create_modality('Геоданные', 'geodata_value')
        coordinates = resource.get('coordinates', {})
        if coordinates:
            lat = self._clean_coordinate(coordinates.get('latitude'))
            lon = self._clean_coordinate(coordinates.get('longitude'))
            if lat is not None and lon is not None:
                geodata_value = GeodataValue(geometry={'type': 'Point', 'coordinates': [lon, lat]})
                value_id = self.modality_repo.save_geodata_value(geodata_value)
            else:
                value_id = None
        else:
            value_id = None

        object_type_name = resource.get('geo_entity_type', 'Географический объект')
        object_type = self._get_or_create_object_type(object_type_name)
        object_obj = self._get_or_create_object(common_name, object_type.id)

        description = resource.get('description')
        if description:
            self._add_object_property(object_obj.id, 'описание', description)

        if coordinates and lat is not None and lon is not None:
            self._add_object_property(object_obj.id, 'координаты', f"{lat}, {lon}")

        for synonym in resource.get('geo_synonyms', []):
            if synonym and synonym != common_name:
                syn = self.synonym_repo.get_or_create(synonym, 'ru', False)
                self.object_repo.add_synonym_link(object_obj.id, syn.id)

        resource_obj = Resource(
            resource_static_id=resource_static_id,
            support_metadata_id=metadata_id
        )
        resource_id = self.resource_repo.save_resource(resource_obj)

        if value_id:
            self.modality_repo.link_resource_value(resource_id, modality.id, value_id)
        self.resource_repo.link_resource_to_object(resource_id, object_obj.id)

        return resource_id

    def _get_biological_name_from_map(self, resource: Dict[str, Any]) -> str:
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

    def _process_weather(self, feature_photo: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        weather_conditions = []
        weather_data = {}

        cloudiness = feature_photo.get('cloudiness')
        if cloudiness and cloudiness != 'Неопределено':
            weather_conditions.append(f"Облачность: {cloudiness}")

        temperature = feature_photo.get('temperature')
        if temperature and temperature != 'Неопределено':
            weather_conditions.append(f"Температура: {temperature}")

        wind = feature_photo.get('wind')
        if wind and wind != 'Неопределено':
            weather_conditions.append(f"Ветер: {wind}")

        precipitation = feature_photo.get('precipitation')
        if precipitation and precipitation != 'Неопределено':
            weather_conditions.append(f"Осадки: {precipitation}")

        weather_text = ', '.join(weather_conditions) if weather_conditions else None
        return weather_text, weather_data

    @staticmethod
    def _clean_coordinate(coord: Any) -> Optional[float]:
        if coord is None:
            return None
        if isinstance(coord, (int, float)):
            return float(coord)
        if isinstance(coord, str):
            try:
                return float(coord.strip())
            except ValueError:
                return None
        return None


@dataclass
class BatchImportUseCase:
    import_resource_use_case: ImportResourceUseCase
    _logger = logging.getLogger(__name__)

    def execute(self, resources: List[Dict[str, Any]], incremental: bool = False) -> ResourceImportResult:
        result = ResourceImportResult()

        for i, resource in enumerate(resources, 1):
            try:
                if i % 100 == 0:
                    self._logger.info(f"Progress: {i}/{len(resources)}")

                resource_id = self.import_resource_use_case.execute(resource, incremental)
                if resource_id:
                    result.success_count += 1
                else:
                    result.skipped_count += 1

            except Exception as e:
                result.error_count += 1
                self._logger.error(f"Error processing resource {i}: {e}", exc_info=True)

        return result