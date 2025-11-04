# utils.py
import json
import hashlib
import logging
from typing import Any, Optional, Tuple
import redis

# Настройка логгера
logger = logging.getLogger(__name__)

# Redis клиент (инициализируется в main приложении)
redis_client = None

def init_redis(host='localhost', port=6379, db=1, decode_responses=True):
    """Инициализация Redis клиента"""
    global redis_client
    redis_client = redis.Redis(
        host=host, 
        port=port, 
        db=db, 
        decode_responses=decode_responses
    )
    try:
        redis_client.ping()
        logger.info("Redis connection established")
    except redis.ConnectionError:
        logger.error("Failed to connect to Redis")
        redis_client = None

def generate_cache_key(params: dict) -> str:
    """
    Создает уникальный MD5 хеш из словаря параметров.
    Ключи сортируются, чтобы порядок не влиял на результат.
    """
    canonical_string = json.dumps(params, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.md5(canonical_string).hexdigest()

def get_cached_result(cache_key: str, debug_info: dict = None) -> Tuple[bool, Optional[Any]]:
    """
    Пытается получить результат из кеша
    
    Returns:
        Tuple[bool, Optional[Any]]: (cache_hit, cached_data)
    """
    if not redis_client:
        return False, None
        
    try:
        cached_result_str = redis_client.get(cache_key)
        if cached_result_str:
            logger.info(f"Cache HIT for key: {cache_key}")
            if debug_info:
                debug_info["cache"] = {"hit": True, "key": cache_key}
            return True, json.loads(cached_result_str)
        else:
            logger.info(f"Cache MISS for key: {cache_key}")
            if debug_info:
                debug_info["cache"] = {"hit": False, "key": cache_key}
            return False, None
    except Exception as e:
        logger.error(f"Redis GET error for key {cache_key}: {e}")
        if debug_info:
            debug_info["cache"] = {"error": str(e)}
        return False, None

def set_cached_result(cache_key: str, result: Any, expire_time: int = 3600) -> bool:
    """
    Сохраняет результат в кеш
    
    Returns:
        bool: True если успешно, False если ошибка
    """
    if not redis_client:
        return False
        
    try:
        redis_client.setex(cache_key, expire_time, json.dumps(result))
        logger.info(f"Cache SET for key: {cache_key} (expire: {expire_time}s)")
        return True
    except Exception as e:
        logger.error(f"Redis SET error for key {cache_key}: {e}")
        return False

def clear_cache_pattern(pattern: str = "cache:*") -> Tuple[bool, int]:
    """
    Очищает кеш по паттерну
    
    Returns:
        Tuple[bool, int]: (success, keys_deleted)
    """
    if not redis_client:
        return False, 0
        
    try:
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache keys with pattern: {pattern}")
            return True, len(keys)
        else:
            logger.info(f"No cache keys found with pattern: {pattern}")
            return True, 0
    except Exception as e:
        logger.error(f"Error clearing cache with pattern {pattern}: {e}")
        return False, 0

def get_cache_stats() -> dict:
    """Статистика кеша"""
    if not redis_client:
        return {"error": "Redis not available"}
        
    try:
        pattern = "cache:*"
        keys = redis_client.keys(pattern)
        
        stats = {
            "total_keys": len(keys),
            "patterns": {}
        }
        
        # Группируем по типам
        for key in keys:
            key_parts = key.split(":")
            if len(key_parts) >= 2:
                cache_type = key_parts[1]
                if cache_type not in stats["patterns"]:
                    stats["patterns"][cache_type] = 0
                stats["patterns"][cache_type] += 1
        
        return stats
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {"error": str(e)}