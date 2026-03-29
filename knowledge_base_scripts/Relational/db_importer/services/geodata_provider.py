"""Geodata provider service."""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class GeodataProvider:
    """Provider for geographical data from geodb.json."""
    
    def __init__(self, geodb_path: Path):
        self._geodb: Dict[str, Any] = self._load_geodb(geodb_path)
    
    def _load_geodb(self, path: Path) -> Dict[str, Any]:
        """Load geodb from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def get_geometry(self, name: str) -> Optional[Dict[str, Any]]:
        """Get geometry by object name."""
        # Direct match
        if name in self._geodb:
            return self._geodb[name].get('geometry')
        
        # Case-insensitive match
        for n, data in self._geodb.items():
            if n.lower() == name.lower():
                return data.get('geometry')
        
        # Partial match (last part of comma-separated name)
        if ',' in name:
            parts = [p.strip() for p in name.split(',')]
            for part in reversed(parts):
                geometry = self.get_geometry(part)
                if geometry:
                    return geometry
        
        return None