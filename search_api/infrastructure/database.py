# search_api/infrastructure/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from ..config import SearchConfig

_engine = None
_SessionLocal = None

def init_db(config: SearchConfig):
    global _engine, _SessionLocal
    db_url = f"postgresql://{config.db_user}:{config.db_password}@{config.db_host}:{config.db_port}/{config.db_name}"
    _engine = create_engine(db_url, pool_pre_ping=True)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

def get_session() -> Session:
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db first.")
    return _SessionLocal()