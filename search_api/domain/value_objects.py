from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class ModalityType(Enum):
    TEXT = "Текст"
    IMAGE = "Изображение"
    GEODATA = "Геоданные"


@dataclass
class MapLinks:
    static: Optional[str] = None
    interactive: Optional[str] = None


@dataclass
class GeoContent:
    geojson: Dict[str, Any]
    geometry_type: str
    map_links: Optional[MapLinks] = None