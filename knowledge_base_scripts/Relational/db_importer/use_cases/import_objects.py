# db_importer/use_cases/import_objects.py

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import logging

from ..domain.entities import (
    Object,
    ObjectType,
    ObjectNameSynonym,
    DbId,
)
from .interfaces import (
    ObjectRepository,
    ObjectTypeRepository,
    SynonymRepository,
    ObjectPropertyRepository,
)


@dataclass
class ImportObjectsUseCase:
    object_repo: ObjectRepository
    object_type_repo: ObjectTypeRepository
    synonym_repo: SynonymRepository
    property_repo: ObjectPropertyRepository

    _logger = logging.getLogger(__name__)

    def execute(self, objects_data: List[Dict[str, Any]]) -> Dict[str, int]:
        result = {'created': 0, 'updated': 0, 'errors': 0}
        object_relations_to_process = []

        for obj_data in objects_data:
            try:
                identificator = obj_data.get('identificator', {})
                db_id_value = identificator.get('db_id')

                if not db_id_value:
                    self._logger.warning(f"Object without db_id, skipping: {obj_data}")
                    result['errors'] += 1
                    continue

                db_id = DbId(db_id_value)

                object_type_name = obj_data.get('type')
                if not object_type_name:
                    self._logger.warning(f"Object without type, skipping: {obj_data}")
                    result['errors'] += 1
                    continue

                object_type = self.object_type_repo.get_or_create(object_type_name)
                existing = self.object_repo.find_by_db_id(str(db_id))

                name_synonyms = obj_data.get('name_synonyms', {})
                primary_name = self._get_primary_name(name_synonyms)

                if existing:
                    existing.object_properties = obj_data.get('properties', {})
                    existing.object_type_id = object_type.id
                    updated_obj = self.object_repo.save(existing)
                    object_id = updated_obj.id
                    result['updated'] += 1
                    self._logger.debug(f"Updated object {db_id}")
                else:
                    object_obj = Object(
                        db_id=db_id,
                        object_type_id=object_type.id,
                        object_properties=obj_data.get('properties', {})
                    )
                    saved_obj = self.object_repo.save(object_obj)
                    object_id = saved_obj.id
                    result['created'] += 1
                    self._logger.debug(f"Created object {db_id}")

                if primary_name:
                    self._process_synonyms(object_id, name_synonyms, primary_name)

                properties = obj_data.get('properties', {})
                if properties:
                    extracted_props = self._extract_properties(properties)
                    for prop_name, values in extracted_props:
                        self.property_repo.add_or_update_property(object_type.id, prop_name, values)

                for relation in obj_data.get('object_relations', []):
                    related_db_id = relation.get('db_id')
                    relation_type = relation.get('type')
                    if related_db_id and relation_type:
                        object_relations_to_process.append((object_id, related_db_id, relation_type))

            except Exception as e:
                self._logger.error(f"Error importing object: {e}", exc_info=True)
                result['errors'] += 1

        for object_id, related_db_id, relation_type in object_relations_to_process:
            try:
                related_obj = self.object_repo.find_by_db_id(related_db_id)
                if related_obj:
                    self.object_repo.link_object_to_object(object_id, related_obj.id, relation_type)
                    self._logger.debug(f"Linked object {object_id} -> {related_obj.id}")
                else:
                    self._logger.warning(f"Related object not found for db_id: {related_db_id}")
            except Exception as e:
                self._logger.error(f"Error linking objects: {e}", exc_info=True)

        self._logger.info(f"Objects import: created={result['created']}, updated={result['updated']}, errors={result['errors']}")
        return result

    def _get_primary_name(self, name_synonyms: Dict[str, List[str]]) -> Optional[str]:
        if name_synonyms.get('ru_names'):
            return name_synonyms['ru_names'][0]
        if name_synonyms.get('scientific_names'):
            return name_synonyms['scientific_names'][0]
        return None

    def _process_synonyms(self, object_id: int, name_synonyms: Dict[str, List[str]], primary_name: str) -> None:
        normalized_primary = primary_name.lower().strip() if primary_name else None

        if normalized_primary:
            synonym = self.synonym_repo.get_or_create(normalized_primary, 'ru')
            self.object_repo.add_synonym_link(object_id, synonym.id)

        for name in name_synonyms.get('ru_names', []):
            if name and name.strip():
                normalized_name = name.strip().lower()
                synonym = self.synonym_repo.get_or_create(normalized_name, 'ru')
                self.object_repo.add_synonym_link(object_id, synonym.id)

        for name in name_synonyms.get('scientific_names', []):
            if name and name.strip():
                normalized_name = name.strip().lower()
                if normalized_name != normalized_primary:
                    synonym = self.synonym_repo.get_or_create(normalized_name, 'sn')
                    self.object_repo.add_synonym_link(object_id, synonym.id)

        for name in name_synonyms.get('en_names', []):
            if name and name.strip():
                normalized_name = name.strip().lower()
                if normalized_name != normalized_primary:
                    synonym = self.synonym_repo.get_or_create(normalized_name, 'en')
                    self.object_repo.add_synonym_link(object_id, synonym.id)

    def _extract_properties(self, properties: Dict[str, Any], max_depth: int = 2) -> List[Tuple[str, List[str]]]:
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

        for key, val in properties.items():
            extract(key, val, 1)

        return result