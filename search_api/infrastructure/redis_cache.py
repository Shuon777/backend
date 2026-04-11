import json
import hashlib
import logging
from typing import Any, Optional, Tuple

import redis

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 1):
        self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def ping(self) -> bool:
        try:
            return self._client.ping()
        except redis.ConnectionError:
            return False

    def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        try:
            value = self._client.get(key)
            if value:
                return True, json.loads(value)
            return False, None
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return False, None

    def set(self, key: str, value: Any, expire_seconds: int = 3600) -> bool:
        try:
            self._client.setex(key, expire_seconds, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False

    @staticmethod
    def generate_key(prefix: str, params: dict) -> str:
        canonical = json.dumps(params, sort_keys=True, ensure_ascii=False).encode('utf-8')
        return f"{prefix}:{hashlib.sha512(canonical).hexdigest()}"