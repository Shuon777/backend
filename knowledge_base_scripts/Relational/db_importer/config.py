import os
from dataclasses import dataclass

@dataclass(frozen=True)
class DatabaseConfig:
    dbname: str
    user: str
    password: str
    host: str
    port: str

    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        return cls(
            dbname=os.getenv('DB_NAME', 'eco'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'Fdf78yh0a4b!'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432')
        )