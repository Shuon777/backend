import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: str
    redis_host: str
    redis_port: int
    redis_db: int
    maps_dir: str
    domain: str

    @classmethod
    def from_env(cls) -> 'SearchConfig':
        return cls(
            db_name=os.getenv('DB_NAME', 'eco'),
            db_user=os.getenv('DB_USER', 'postgres'),
            db_password=os.getenv('DB_PASSWORD', 'Fdf78yh0a4b!'),
            db_host=os.getenv('DB_HOST', 'localhost'),
            db_port=os.getenv('DB_PORT', '5432'),
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', '6379')),
            redis_db=int(os.getenv('REDIS_DB', '1')),
            maps_dir=os.getenv('MAPS_DIR', '/app/maps'),
            domain=os.getenv('DOMAIN', 'http://localhost:5555')
        )