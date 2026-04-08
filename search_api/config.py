# search_api/config.py
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: str

    @classmethod
    def from_env(cls) -> 'SearchConfig':
        return cls(
            db_name=os.getenv('DB_NAME', 'eco'),
            db_user=os.getenv('DB_USER', 'postgres'),
            db_password=os.getenv('DB_PASSWORD', 'Fdf78yh0a4b!'),
            db_host=os.getenv('DB_HOST', 'localhost'),
            db_port=os.getenv('DB_PORT', '5432')
        )