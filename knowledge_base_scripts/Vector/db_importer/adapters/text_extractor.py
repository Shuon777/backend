# adapters/text_extractor.py
from typing import Dict, Any

from ..domain.entities import ResourceForIndexing
from ..domain.interfaces import TextExtractor


class NewResourceTextExtractor(TextExtractor):
    def __init__(self, objects_mapper):
        self._objects_mapper = objects_mapper

    def extract(self, resource: ResourceForIndexing) -> str:
        if resource.modality_type != "Текст":
            return ""
        if resource.title and resource.title.startswith("Описание изображения:"):
            return ""

        parts = []
        if resource.title:
            parts.append(resource.title)

        for rel in resource.object_relations:
            db_id = rel.get('db_id')
            if db_id:
                names = self._objects_mapper.get_object_names(db_id)
                if names:
                    parts.append(f"Объект {db_id}: {names}")

        structured = resource.modality_value.get("structured_data", {})
        content = structured.get("content", "")
        if content:
            parts.append(content)

        return ".\n".join(parts).strip()