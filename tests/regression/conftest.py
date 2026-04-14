# tests/regression/conftest.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'knowledge_base_scripts' / 'Relational'))

from tests.conftest import *