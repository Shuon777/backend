from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ResourceForIndexing:
    resource_id: str
    title: Optional[str]
    modality_type: str
    modality_value: Dict[str, Any]
    object_relations: list


@dataclass
class TextChunk:
    text: str
    metadata: Dict[str, Any]