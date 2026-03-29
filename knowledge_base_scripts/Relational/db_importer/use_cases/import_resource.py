"""Import resource use cases."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import json
import logging

from ..domain.entities import (
    Resource,
    ResourceImportResult,
    SupportMetadata,
    CanonicalId,
    ObjectDescription,
    ObjectSynonym,
    PropertyValue,
    ObjectProperty,
    BibliographicData,
    GenerationData,
    TextModality,
    ImageModality,
    GeodataModality,
    ResourceType,
)
from .interfaces import (
    ResourceRepository,
    ObjectDescriptionRepository,
    PropertyValueRepository,
    ModalityRepository,
    BibliographicRepository,
    GenerationRepository,
    SupportMetadataRepository,
    SpeciesNameNormalizer,
)


@dataclass
class ImportResourceUseCase:
    """Use case for importing a single resource."""
    
    resource_repo: ResourceRepository
    object_repo: ObjectDescriptionRepository
    property_value_repo: PropertyValueRepository
    modality_repo: ModalityRepository
    bibliographic_repo: BibliographicRepository
    generation_repo: GenerationRepository
    metadata_repo: SupportMetadataRepository
    species_normalizer: SpeciesNameNormalizer
    
    _logger = logging.getLogger(__name__)
    
    def execute(self, resource_data: Dict[str, Any], incremental: bool = False) -> Optional[int]:
        """Execute resource import."""
        
        # Check for duplicate if incremental
        if incremental:
            resource_hash = self._calculate_hash(resource_data)
            if self.resource_repo.resource_exists_by_hash(resource_hash):
                return None
        else:
            resource_hash = None
        
        # Process based on resource type
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
        """Calculate resource hash for deduplication."""
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
        """Extract resource title."""
        common = resource['identificator'].get('name', {}).get('common')
        if common:
            return common
        original = resource.get('access_options', {}).get('original_title')
        if original:
            return original
        return resource['identificator'].get('id', 'Без названия')
    
    def _get_reliability(self, source: Optional[str]) -> str:
        """Determine reliability level from source."""
        if not source:
            return "общедоступная"
        
        src_low = source.lower()
        if "национальный парк" in src_low or "заповедник" in src_low:
            return "профильная организация"
        if "ai generation" in src_low or "википедия" in src_low:
            return "общедоступная"
        return "профильная организация"
    
    def _create_bibliographic(self, resource: Dict[str, Any], date: Optional[str] = None) -> int:
        """Create bibliographic data."""
        access_options = resource.get('access_options', {})
        name_info = resource.get('identificator', {}).get('name', {})
        
        bibliographic = BibliographicData(
            author=access_options.get('author'),
            date=date,
            source=name_info.get('source'),
            reliability=self._get_reliability(name_info.get('source'))
        )
        return self.bibliographic_repo.get_or_create(bibliographic)
    
    def _create_support_metadata(self, resource: Dict[str, Any], 
                                  resource_hash: Optional[str] = None) -> int:
        """Create support metadata."""
        metadata = SupportMetadata.from_resource(resource, resource_hash)
        return self.metadata_repo.get_or_create(metadata)
    
    def _get_or_create_object(self, name: str, object_type: str) -> ObjectDescription:
        """Get or create object description."""
        canonical_id = CanonicalId.from_name_and_type(name, object_type)
        
        existing = self.object_repo.find_by_canonical_id(str(canonical_id), object_type)
        if existing:
            return existing
        
        object_desc = ObjectDescription(
            canonical_id=canonical_id,
            object_type=object_type
        )
        object_desc = self.object_repo.save(object_desc)
        
        # Add primary synonym
        synonym = ObjectSynonym(
            object_description_id=object_desc.id,
            synonym=name,
            is_primary=True
        )
        self.object_repo.add_synonym(synonym)
        
        return object_desc
    
    def _add_object_property(self, object_id: int, object_type: str, 
                             property_name: str, value: str) -> None:
        """Add property to object."""
        property_value = self.property_value_repo.get_or_create(value)
        
        property_obj = ObjectProperty(
            object_description_id=object_id,
            property_name=property_name,
            object_type=object_type,
            property_value=property_value
        )
        self.object_repo.add_property(property_obj)
    
    def _import_image(self, resource: Dict[str, Any], resource_hash: Optional[str] = None) -> Optional[int]:
        """Import image resource."""
        
        # Create base components
        bibliographic_id = self._create_bibliographic(resource)
        generation_id = self.generation_repo.get_or_create(GenerationData())
        metadata_id = self._create_support_metadata(resource, resource_hash)
        
        # Create modality
        modality_id = self.modality_repo.get_or_create_modality('Изображение')
        
        access_options = resource.get('access_options', {})
        feature_photo = resource.get('featurePhoto', {})
        
        image_modality = ImageModality(
            modality_id=modality_id,
            url=access_options.get('source_url'),
            file_path=access_options.get('file_path') or feature_photo.get('file_path')
        )
        self.modality_repo.save_image_modality(image_modality)
        
        # Create resource
        resource_obj = Resource(
            modality_id=modality_id,
            bibliographic_id=bibliographic_id,
            generation_id=generation_id,
            support_metadata_id=metadata_id
        )
        resource_id = self.resource_repo.save_resource(resource_obj)
        
        # Process object if exists
        name_info = resource.get('identificator', {}).get('name', {})
        common_name = name_info.get('common')
        
        if common_name:
            normalized_name = self.species_normalizer.normalize(common_name)
            information_subtype = resource.get('information_subtype')
            object_type = information_subtype or resource.get('information_type', 'Изображение')
            
            object_desc = self._get_or_create_object(normalized_name, object_type)
            self.resource_repo.link_resource_to_object(resource_id, object_desc.id)
            
            # Add classification properties
            classification_info = feature_photo.get('classification_info', {})
            if classification_info:
                result_info = classification_info.get('result', {})
                for key, value in result_info.items():
                    if value and key not in ['source']:
                        self._add_object_property(object_desc.id, object_type, key, str(value))
            
            # Add weather conditions
            weather_text, _ = self._process_weather(feature_photo)
            if weather_text:
                self._add_object_property(object_desc.id, object_type, 'погодные условия', weather_text)
            
            # Add coordinates
            location = feature_photo.get('location', {})
            if location:
                lat = self._clean_coordinate(location.get('latitude'))
                lon = self._clean_coordinate(location.get('longitude'))
                if lat is not None and lon is not None:
                    self._add_object_property(object_desc.id, object_type, 'координаты', f"{lat}, {lon}")
        
        return resource_id
    
    def _import_text(self, resource: Dict[str, Any], resource_hash: Optional[str] = None) -> Optional[int]:
        """Import text resource."""
        
        bibliographic_id = self._create_bibliographic(resource)
        generation_id = self.generation_repo.get_or_create(GenerationData())
        metadata_id = self._create_support_metadata(resource, resource_hash)
        
        modality_id = self.modality_repo.get_or_create_modality('Текст')
        
        title = self._get_title(resource)
        text_content = {
            'content': resource.get('content', ''),
            'structured_data': resource.get('structured_data', {}),
            'title': title,
            'brief_annotation': resource.get('brief_annotation', '')
        }
        
        text_modality = TextModality(
            modality_id=modality_id,
            content=text_content
        )
        self.modality_repo.save_text_modality(text_modality)
        
        resource_obj = Resource(
            modality_id=modality_id,
            bibliographic_id=bibliographic_id,
            generation_id=generation_id,
            support_metadata_id=metadata_id
        )
        resource_id = self.resource_repo.save_resource(resource_obj)
        
        # Create object
        object_type = resource.get('information_type', 'Текстовый ресурс')
        object_desc = self._get_or_create_object(title, object_type)
        self.resource_repo.link_resource_to_object(resource_id, object_desc.id)
        
        # Add annotation
        brief_annotation = resource.get('brief_annotation')
        if brief_annotation:
            self._add_object_property(object_desc.id, object_type, 'аннотация', brief_annotation)
        
        return resource_id
    
    def _import_map(self, resource: Dict[str, Any], resource_hash: Optional[str] = None) -> Optional[int]:
        """Import map resource."""
        
        bibliographic_id = self._create_bibliographic(resource)
        generation_id = self.generation_repo.get_or_create(GenerationData())
        metadata_id = self._create_support_metadata(resource, resource_hash)
        
        modality_id = self.modality_repo.get_or_create_modality('Геоданные')
        
        # Get biological object name
        common_name = self._get_biological_name_from_map(resource)
        if common_name:
            common_name = self.species_normalizer.normalize(common_name)
        
        scientific_name = resource.get('plant_latin_name') or resource.get('animal_latin_name')
        information_subtype = resource.get('information_subtype')
        object_type = information_subtype or 'Биологический объект'
        
        object_desc = self._get_or_create_object(common_name, object_type)
        
        if scientific_name:
            synonym = ObjectSynonym(
                object_description_id=object_desc.id,
                synonym=scientific_name,
                language='la'
            )
            self.object_repo.add_synonym(synonym)
        
        resource_obj = Resource(
            modality_id=modality_id,
            bibliographic_id=bibliographic_id,
            generation_id=generation_id,
            support_metadata_id=metadata_id
        )
        resource_id = self.resource_repo.save_resource(resource_obj)
        self.resource_repo.link_resource_to_object(resource_id, object_desc.id)
        
        return resource_id
    
    def _import_geographical_object(self, resource: Dict[str, Any], 
                                     resource_hash: Optional[str] = None) -> Optional[int]:
        """Import geographical object resource."""
        
        identificator = resource.get('identificator', {})
        name_info = identificator.get('name', {})
        common_name = name_info.get('common')
        
        if not common_name:
            return None
        
        bibliographic_id = self._create_bibliographic(resource)
        generation_id = self.generation_repo.get_or_create(GenerationData())
        metadata_id = self._create_support_metadata(resource, resource_hash)
        
        modality_id = self.modality_repo.get_or_create_modality('Геоданные')
        
        object_type = resource.get('geo_entity_type', 'Географический объект')
        object_desc = self._get_or_create_object(common_name, object_type)
        
        # Add description
        description = resource.get('description')
        if description:
            self._add_object_property(object_desc.id, object_type, 'описание', description)
        
        # Add coordinates
        coordinates = resource.get('coordinates', {})
        if coordinates:
            lat = self._clean_coordinate(coordinates.get('latitude'))
            lon = self._clean_coordinate(coordinates.get('longitude'))
            if lat is not None and lon is not None:
                self._add_object_property(object_desc.id, object_type, 'координаты', f"{lat}, {lon}")
        
        # Add synonyms
        for synonym in resource.get('geo_synonyms', []):
            if synonym and synonym != common_name:
                synonym_obj = ObjectSynonym(
                    object_description_id=object_desc.id,
                    synonym=synonym
                )
                self.object_repo.add_synonym(synonym_obj)
        
        resource_obj = Resource(
            modality_id=modality_id,
            bibliographic_id=bibliographic_id,
            generation_id=generation_id,
            support_metadata_id=metadata_id
        )
        resource_id = self.resource_repo.save_resource(resource_obj)
        self.resource_repo.link_resource_to_object(resource_id, object_desc.id)
        
        return resource_id
    
    def _get_biological_name_from_map(self, resource: Dict[str, Any]) -> str:
        """Extract biological object name from map."""
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
        """Process weather conditions."""
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
        """Clean coordinate value."""
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
    """Use case for batch resource import."""
    
    import_resource_use_case: ImportResourceUseCase
    _logger = logging.getLogger(__name__)
    
    def execute(self, resources: List[Dict[str, Any]], incremental: bool = False) -> ResourceImportResult:
        """Execute batch import."""
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