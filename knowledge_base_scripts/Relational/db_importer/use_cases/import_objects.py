from dataclasses import dataclass
from typing import Dict, Any, List, Optional
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
)


@dataclass
class ImportObjectsUseCase:
    object_repo: ObjectRepository
    object_type_repo: ObjectTypeRepository
    synonym_repo: SynonymRepository
    
    _logger = logging.getLogger(__name__)
    
    def execute(self, objects_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Import objects from objects.json
        Returns dict with counts: {'created': n, 'updated': n, 'errors': n}
        """
        result = {'created': 0, 'updated': 0, 'errors': 0}
        
        # Список для отложенной обработки связей
        object_relations_to_process = []  # (object_id, related_db_id, relation_type)
        
        for obj_data in objects_data:
            try:
                # 1. Получаем db_id из identificator
                identificator = obj_data.get('identificator', {})
                db_id_value = identificator.get('db_id')
                
                if not db_id_value:
                    self._logger.warning(f"Object without db_id, skipping: {obj_data}")
                    result['errors'] += 1
                    continue
                
                # Используем существующий db_id (уже не генерируем)
                db_id = DbId(db_id_value)
                
                # 2. Получаем или создаем тип объекта
                object_type_name = obj_data.get('type')
                if not object_type_name:
                    self._logger.warning(f"Object without type, skipping: {obj_data}")
                    result['errors'] += 1
                    continue
                
                object_type = self.object_type_repo.get_or_create(object_type_name)
                
                # 3. Ищем существующий объект по db_id
                existing = self.object_repo.find_by_db_id(str(db_id))
                
                # 4. Получаем основное имя для синонимов
                name_synonyms = obj_data.get('name_synonyms', {})
                primary_name = None
                
                if name_synonyms.get('ru_names'):
                    primary_name = name_synonyms['ru_names'][0]
                elif name_synonyms.get('scientific_names'):
                    primary_name = name_synonyms['scientific_names'][0]
                
                # 5. Создаем или обновляем объект
                if existing:
                    # Обновляем существующий объект
                    existing.object_properties = obj_data.get('properties', {})
                    # Сохраняем оригинальный object_type_id (может измениться)
                    existing.object_type_id = object_type.id
                    updated_obj = self.object_repo.save(existing)
                    object_id = updated_obj.id
                    result['updated'] += 1
                    self._logger.debug(f"Updated object {db_id} (type: {object_type_name})")
                else:
                    # Создаем новый объект
                    object_obj = Object(
                        db_id=db_id,
                        object_type_id=object_type.id,
                        object_properties=obj_data.get('properties', {})
                    )
                    saved_obj = self.object_repo.save(object_obj)
                    object_id = saved_obj.id
                    result['created'] += 1
                    self._logger.debug(f"Created object {db_id} (type: {object_type_name})")
                
                # 6. Обрабатываем синонимы
                if primary_name:
                    self._process_synonyms(object_id, name_synonyms, primary_name)
                
                # 7. Сохраняем связи для отложенной обработки
                for relation in obj_data.get('object_relations', []):
                    related_db_id = relation.get('db_id')
                    relation_type = relation.get('type')
                    if related_db_id and relation_type:
                        object_relations_to_process.append((object_id, related_db_id, relation_type))
                
            except Exception as e:
                self._logger.error(f"Error importing object: {e}", exc_info=True)
                result['errors'] += 1
        
        # Второй проход: обрабатываем связи между объектами
        for object_id, related_db_id, relation_type in object_relations_to_process:
            try:
                related_obj = self.object_repo.find_by_db_id(related_db_id)
                if related_obj:
                    self.object_repo.link_object_to_object(object_id, related_obj.id, relation_type)
                    self._logger.debug(f"Linked object {object_id} -> {related_obj.id} ({relation_type})")
                else:
                    self._logger.warning(f"Related object not found for db_id: {related_db_id}")
            except Exception as e:
                self._logger.error(f"Error linking objects: {e}", exc_info=True)
        
        self._logger.info(f"Objects import completed: created={result['created']}, updated={result['updated']}, errors={result['errors']}")
        return result
    
    def _process_synonyms(self, object_id: int, name_synonyms: Dict[str, List[str]], primary_name: str) -> None:
        """Process and link all synonyms to object (case-insensitive, all lowercase)"""
        
        # Нормализуем primary_name для сравнения
        normalized_primary = primary_name.lower().strip() if primary_name else None
        
        # Добавляем primary_name как синоним (если есть)
        if normalized_primary:
            synonym = self.synonym_repo.get_or_create(normalized_primary, 'ru')
            self.object_repo.add_synonym_link(object_id, synonym.id)
        
        # Обрабатываем русские названия
        for name in name_synonyms.get('ru_names', []):
            if name and name.strip():
                normalized_name = name.strip().lower()
                synonym = self.synonym_repo.get_or_create(normalized_name, 'ru')
                self.object_repo.add_synonym_link(object_id, synonym.id)
        
        # Обрабатываем научные названия
        for name in name_synonyms.get('scientific_names', []):
            if name and name.strip():
                normalized_name = name.strip().lower()
                # Проверяем, не дублирует ли это primary_name
                if normalized_name != normalized_primary:
                    synonym = self.synonym_repo.get_or_create(normalized_name, 'sn')
                    self.object_repo.add_synonym_link(object_id, synonym.id)
        
        # Обрабатываем английские названия
        for name in name_synonyms.get('en_names', []):
            if name and name.strip():
                normalized_name = name.strip().lower()
                if normalized_name != normalized_primary:
                    synonym = self.synonym_repo.get_or_create(normalized_name, 'en')
                    self.object_repo.add_synonym_link(object_id, synonym.id)