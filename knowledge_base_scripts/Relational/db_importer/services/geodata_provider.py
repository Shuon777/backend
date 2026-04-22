"""Geodata provider service."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class GeodataProvider:
    """Provider for geographical data from geodb.json."""
    
    _GEOMETRY_TYPE_MAP = {
        'точка': 'Point',
        'Point': 'Point',
        'линия': 'LineString',
        'LineString': 'LineString',
        'полигон': 'Polygon',
        'Polygon': 'Polygon',
        'мультиполигон': 'MultiPolygon',
        'MultiPolygon': 'MultiPolygon',
        'мультиточка': 'MultiPoint',
        'MultiPoint': 'MultiPoint',
        'мультилиния': 'MultiLineString',
        'MultiLineString': 'MultiLineString',
    }
    
    def __init__(self, geodb_path: Path):
        self._geodb: Dict[str, Any] = self._load_geodb(geodb_path)
    
    def _load_geodb(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load geodb.json: {e}")
            return {}
    
    def _normalize_geometry_type(self, geom_type: str) -> str:
        if not geom_type:
            return 'Point'
        normalized = self._GEOMETRY_TYPE_MAP.get(geom_type, geom_type)
        return normalized
    
    def get_geometry(self, geodb_id: str) -> Optional[Tuple[Dict[str, Any], str]]:
        if geodb_id in self._geodb:
            data = self._geodb[geodb_id]
            geometry = data.get('geometry')
            if geometry:
                geom_type = geometry.get('type', 'Point')
                normalized_type = self._normalize_geometry_type(geom_type)
                return (geometry, normalized_type)
        
        for key, data in self._geodb.items():
            if key.lower() == geodb_id.lower():
                geometry = data.get('geometry')
                if geometry:
                    geom_type = geometry.get('type', 'Point')
                    normalized_type = self._normalize_geometry_type(geom_type)
                    return (geometry, normalized_type)
        