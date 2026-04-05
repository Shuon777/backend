"""Geodata provider service."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class GeodataProvider:
    """Provider for geographical data from geodb.json."""
    
    def __init__(self, geodb_path: Path):
        self._geodb: Dict[str, Any] = self._load_geodb(geodb_path)
    
    def _load_geodb(self, path: Path) -> Dict[str, Any]:
        """Load geodb from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load geodb.json: {e}")
            return {}
    
    def get_geometry(self, geodb_id: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        Get geometry by geodb_id.
        Returns tuple (geometry_dict, geometry_type) or None.
        """
        # Direct match by key (geodb_id is the key in JSON)
        if geodb_id in self._geodb:
            data = self._geodb[geodb_id]
            geometry = data.get('geometry')
            if geometry:
                geometry_type = geometry.get('type')
                return (geometry, geometry_type)
        
        # Try case-insensitive match
        for key, data in self._geodb.items():
            if key.lower() == geodb_id.lower():
                geometry = data.get('geometry')
                if geometry:
                    geometry_type = geometry.get('type')
                    return (geometry, geometry_type)
        
        return None