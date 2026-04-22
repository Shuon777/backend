# db_importer/use_cases/import_resources.py

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import logging
import hashlib
import json
from pathlib import Path

from ..domain.entities import (
    Resource,
    SupportMetadata,
    BibliographicData,
    CreationData,
    ResourceStatic,
    TextValue,
    ImageValue,
    GeodataValue,
)
from .interfaces import (
    ResourceRepository,
    ObjectRepository,
    ResourceStaticRepository,
    SupportMetadataRepository,
    BibliographicRepository,
    CreationRepository,
    ModalityRepository,
    GeodataProvider,
    ResourceFeatureRepository,
)


@dataclass
class ImportResourcesUseCase:
    resource_repo: ResourceRepository
    object_repo: ObjectRepository
    resource_static_repo: ResourceStaticRepository
    metadata_repo: SupportMetadataRepository
    bibliographic_repo: BibliographicRepository
    creation_repo: CreationRepository
    modality_repo: ModalityRepository
    geodata_provider: GeodataProvider
    feature_repo: ResourceFeatureRepository
    missing_geometry_file: Path = Path(__file__).parent.parent.parent / 'missing_geometry.json'

    _logger = logging.getLogger(__name__)

    def execute(self, resources_data: List[Dict[str, Any]], incremental: bool = False) -> Dict[str, int]:
        self._reset_missing_geometry_file()
        result = {'success': 0, 'skipped': 0, 'errors': 0}
        resource_relations_to_process = []

        for i, resource_data in enumerate(resources_data, 1):
            try:
                if incremental:
                    resource_hash = self._calculate_hash(resource_data)
                    if self.resource_repo.resource_exists_by_hash(resource_hash):
                        result['skipped'] += 1
                        continue
                else:
                    resource_hash = None

                resource_id = self._import_single_resource(resource_data, resource_hash)

                if resource_id:
                    result['success'] += 1
                    for relation in resource_data.get('resource_relations', []):
                        related_id = relation.get('id')
                        relation_type = relation.get('type')
                        if related_id and relation_type:
                            resource_relations_to_process.append((resource_id, related_id, relation_type))
                else:
                    result['errors'] += 1

            except Exception as e:
                self._logger.error(f"Error importing resource {i}: {e}", exc_info=True)
                result['errors'] += 1

        for resource_id, related_id, relation_type in resource_relations_to_process:
            try:
                related_resource_id = self.resource_repo.find_resource_by_text_id(related_id)
                if related_resource_id:
                    self.resource_repo.link_resource_to_resource(resource_id, related_resource_id, relation_type)
                else:
                    self._logger.warning(f"Related resource not found: {related_id}")
            except Exception as e:
                self._logger.error(f"Error linking resources: {e}", exc_info=True)

        self._logger.info(f"Resources import: success={result['success']}, skipped={result['skipped']}, errors={result['errors']}")
        return result

    def _reset_missing_geometry_file(self) -> None:
        try:
            self.missing_geometry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.missing_geometry_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._logger.error(f"Failed to reset missing_geometry.json: {e}")

    def _add_missing_geometry(self, geodb_id: str) -> None:
        try:
            existing_data = []
            if self.missing_geometry_file.exists():
                with open(self.missing_geometry_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

            if geodb_id not in existing_data:
                existing_data.append(geodb_id)
                with open(self.missing_geometry_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._logger.error(f"Failed to add missing geometry: {e}")

    def _import_single_resource(self, resource_data: Dict[str, Any], resource_hash: Optional[str] = None) -> Optional[int]:
        title = resource_data.get('title')
        uri = resource_data.get('identificator', {}).get('uri')
        text_id = resource_data.get('identificator', {}).get('id')

        bibliographic_data = resource_data.get('bibliographic', {})
        author = bibliographic_data.get('author')
        source = bibliographic_data.get('source')
        reliability_level = bibliographic_data.get('reliability_level')

        author_id = self.bibliographic_repo.get_or_create_author(author) if author else None
        source_id = self.bibliographic_repo.get_or_create_source(source) if source else None
        reliability_id = self.bibliographic_repo.get_or_create_reliability_level(reliability_level) if reliability_level else None

        bibliographic = BibliographicData(
            author_id=author_id,
            date=None,
            source_id=source_id,
            reliability_level_id=reliability_id
        )
        bibliographic_id = self.bibliographic_repo.get_or_create(bibliographic)

        creation_data = resource_data.get('creation', {})
        creation = CreationData(
            creation_type=creation_data.get('creation_type'),
            creation_tool=creation_data.get('creation_tool'),
            creation_params=None
        )
        creation_id = self.creation_repo.get_or_create(creation)

        resource_static = ResourceStatic(
            static_id=None,
            bibliographic_id=bibliographic_id,
            creation_id=creation_id
        )
        resource_static_id = self.resource_static_repo.get_or_create(resource_static)

        support_metadata_data = resource_data.get('support_metadata', {})
        metadata_params = {
            'external_id': support_metadata_data.get('external_id'),
            'external_url': support_metadata_data.get('external_url'),
            'question': support_metadata_data.get('question')
        }
        if resource_hash:
            metadata_params['resource_hash'] = resource_hash
        metadata = SupportMetadata(parameters=metadata_params)
        metadata_id = self.metadata_repo.get_or_create(metadata)

        features = resource_data.get('features', [])
        features_json = self._build_features_json(features)

        resource = Resource(
            title=title,
            uri=uri,
            features=features_json,
            text_id=text_id,
            resource_static_id=resource_static_id,
            support_metadata_id=metadata_id
        )
        resource_id = self.resource_repo.save_resource(resource)

        modality_data = resource_data.get('modality', {})
        modality_type = modality_data.get('type')
        modality_value = modality_data.get('value', {})

        modality_id = None
        if modality_type:
            modality_id = self._get_or_create_modality_by_type(modality_type)
            self._process_modality(resource_id, modality_type, modality_value)

        if features and modality_id:
            extracted = self._extract_features(features)
            for feat_name, values in extracted:
                self.feature_repo.add_or_update_feature(modality_id, feat_name, values)

        for relation in resource_data.get('object_relations', []):
            object_db_id = relation.get('db_id')
            relation_type = relation.get('type')
            if object_db_id:
                obj = self.object_repo.find_by_db_id_only(object_db_id)
                if obj:
                    self.resource_repo.link_resource_to_object(resource_id, obj.id, relation_type)

        return resource_id

    def _get_or_create_modality_by_type(self, modality_type: str) -> int:
        if modality_type in ("Текст", "Text"):
            modality = self.modality_repo.get_or_create_modality('Текст', 'text_value')
        elif modality_type in ("Изображение", "Image"):
            modality = self.modality_repo.get_or_create_modality('Изображение', 'image_value')
        elif modality_type in ("Геоданные", "Картографическая информация"):
            modality = self.modality_repo.get_or_create_modality('Геоданные', 'geodata_value')
        else:
            modality = self.modality_repo.get_or_create_modality(modality_type, 'text_value')
        return modality.id

    def _process_modality(self, resource_id: int, modality_type: str, modality_value: Dict[str, Any]) -> None:
        if modality_type in ("Текст", "Text"):
            modality = self.modality_repo.get_or_create_modality('Текст', 'text_value')
            structured_data = modality_value.get('structured_data', {})
            text_value = TextValue(structured_data=structured_data)
            value_id = self.modality_repo.save_text_value(text_value)
            self.modality_repo.link_resource_value(resource_id, modality.id, value_id)

        elif modality_type in ("Изображение", "Image"):
            modality = self.modality_repo.get_or_create_modality('Изображение', 'image_value')
            image_value = ImageValue(
                url=modality_value.get('url'),
                file_path=modality_value.get('file_path'),
                format=modality_value.get('format')
            )
            value_id = self.modality_repo.save_image_value(image_value)
            self.modality_repo.link_resource_value(resource_id, modality.id, value_id)

        elif modality_type in ("Геоданные", "Картографическая информация"):
            modality = self.modality_repo.get_or_create_modality('Геоданные', 'geodata_value')
            geodb_id = modality_value.get('geodb_id')
            if geodb_id:
                geometry_data = self.geodata_provider.get_geometry(geodb_id)
                if geometry_data:
                    geometry, normalized_type = geometry_data
                    geodata_value = GeodataValue(
                        geometry=geometry,
                        geometry_type=normalized_type
                    )
                    value_id = self.modality_repo.save_geodata_value(geodata_value)
                    self.modality_repo.link_resource_value(resource_id, modality.id, value_id)
                else:
                    self._add_missing_geometry(geodb_id)
            else:
                self._logger.warning("Geodata modality without geodb_id")

    def _build_features_json(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = {}
        for feature in features:
            name = feature.get('name')
            value = feature.get('value')
            if name and value is not None:
                result[name] = value
        return result

    def _calculate_hash(self, resource: Dict[str, Any]) -> str:
        data = {
            'title': resource.get('title'),
            'identificator': resource.get('identificator'),
            'bibliographic': resource.get('bibliographic'),
            'modality': resource.get('modality'),
            'creation': resource.get('creation'),
            'features': resource.get('features')
        }
        return hashlib.md5(
            json.dumps(data, sort_keys=True, ensure_ascii=False).encode('utf-8')
        ).hexdigest()

    def _extract_features(self, features: List[Dict[str, Any]], max_depth: int = 2) -> List[Tuple[str, List[str]]]:
        result = []

        def extract(prefix: str, value: Any, depth: int) -> None:
            if depth > max_depth:
                return
            if isinstance(value, dict):
                for k, v in value.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    extract(new_prefix, v, depth + 1)
            elif isinstance(value, list):
                str_values = [str(item) for item in value if item is not None]
                if str_values:
                    result.append((prefix, str_values))
            else:
                if value is not None:
                    result.append((prefix, [str(value)]))

        for feat in features:
            name = feat.get('name')
            value = feat.get('value')
            if not name:
                continue
            extract(name, value, 1)

        return result