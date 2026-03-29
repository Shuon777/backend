"""Species name normalization service."""

import json
from pathlib import Path
from typing import Dict, List

from ..use_cases.interfaces import SpeciesNameNormalizer


class JsonSpeciesNormalizer(SpeciesNameNormalizer):
    """Species name normalizer based on JSON file."""
    
    def __init__(self, synonyms_path: Path):
        self._synonyms: Dict[str, List[str]] = self._load_synonyms(synonyms_path)
    
    def _load_synonyms(self, path: Path) -> Dict[str, List[str]]:
        """Load synonyms from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('biological_entity', {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def normalize(self, name: str) -> str:
        """Normalize species name."""
        if not name:
            return name
        
        name_lower = name.strip().lower()
        
        for main_name, synonyms in self._synonyms.items():
            if name_lower == main_name.lower():
                return main_name
            for synonym in synonyms:
                if name_lower == synonym.lower():
                    return main_name
        
        return name