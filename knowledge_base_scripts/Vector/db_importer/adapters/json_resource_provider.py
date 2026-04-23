# adapters/json_resource_provider.py
import json
from pathlib import Path
from typing import List, Dict, Any

from ..domain.entities import ResourceForIndexing
from ..domain.interfaces import ResourceProvider


class JsonResourceProvider(ResourceProvider):
    def __init__(self, resources_path: Path):
        self._resources_path = resources_path
        self._objects_map = self._load_objects()

    def _load_objects(self) -> Dict[str, Dict[str, Any]]:
        objects_path = self._resources_path.parent / "objects.json"
        if not objects_path.exists():
            return {}
        with open(objects_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        objects = data.get('objects', [])
        return {obj['identificator']['db_id']: obj for obj in objects}

    def get_object_names(self, db_id: str) -> str:
        obj = self._objects_map.get(db_id)
        if not obj:
            return ""
        names = obj.get('name_synonyms', {})
        ru = ', '.join(names.get('ru_names', []))
        sc = ', '.join(names.get('scientific_names', []))
        en = ', '.join(names.get('en_names', []))
        parts = []
        if ru:
            parts.append(f"русские названия: {ru}")
        if sc:
            parts.append(f"научные названия: {sc}")
        if en:
            parts.append(f"английские названия: {en}")
        return '; '.join(parts)

    def get_resources(self) -> List[ResourceForIndexing]:
        with open(self._resources_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        resources = data.get('resources', [])
        result = []
        for r in resources:
            ident = r.get('identificator', {})
            modality = r.get('modality', {})
            result.append(ResourceForIndexing(
                resource_id=ident.get('id', ''),
                title=r.get('title'),
                modality_type=modality.get('type', ''),
                modality_value=modality.get('value', {}),
                object_relations=r.get('object_relations', [])
            ))
        return result