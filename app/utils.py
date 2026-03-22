import json
import logging
import hashlib
import time
from typing import Optional, Dict, Any, Tuple
import redis

logger = logging.getLogger(__name__)

# Глобальный клиент Redis
redis_client = None

def init_redis(host: str, port: str, db: str, decode_responses: bool = True):
    """Инициализация Redis клиента"""
    global redis_client
    try:
        redis_client = redis.Redis(
            host=host,
            port=int(port),
            db=int(db),
            decode_responses=decode_responses
        )
        redis_client.ping()
        logger.info(f"✅ Redis подключен: {host}:{port}/{db}")
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к Redis: {e}")
        redis_client = None

def generate_cache_key(params: Dict[str, Any]) -> str:
    """Генерация уникального ключа кэша на основе параметров"""
    # Сортируем ключи для стабильности
    sorted_params = sorted(params.items())
    # Преобразуем в строку и хэшируем
    params_str = json.dumps(sorted_params, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(params_str.encode('utf-8')).hexdigest()

def get_cached_result(redis_key: str, debug_info: Optional[Dict] = None) -> Tuple[bool, Optional[Dict]]:
    """Получение результата из кэша"""
    if not redis_client:
        if debug_info is not None:
            debug_info["cache_status"] = "redis_not_available"
        return False, None
    
    try:
        cached = redis_client.get(redis_key)
        if cached:
            result = json.loads(cached)
            if debug_info is not None:
                debug_info["cache_status"] = "hit"
                debug_info["cache_hit_time"] = time.time()
            return True, result
        else:
            if debug_info is not None:
                debug_info["cache_status"] = "miss"
            return False, None
    except Exception as e:
        logger.warning(f"Ошибка получения из кэша: {e}")
        if debug_info is not None:
            debug_info["cache_status"] = f"error: {str(e)}"
        return False, None

def set_cached_result(redis_key: str, data: Dict, expire_time: int = 3600):
    """Сохранение результата в кэш"""
    if not redis_client:
        return
    
    try:
        # Преобразуем numpy типы в стандартные Python типы
        serializable_data = json.loads(json.dumps(data, default=str))
        redis_client.setex(redis_key, expire_time, json.dumps(serializable_data))
        logger.debug(f"Сохранено в кэш: {redis_key} (expire: {expire_time}s)")
    except Exception as e:
        logger.warning(f"Ошибка сохранения в кэш: {e}")

def clear_cache_pattern(pattern: str = "cache:*"):
    """Очистка кэша по паттерну"""
    if not redis_client:
        return
    
    try:
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Очищено {len(keys)} ключей по паттерну {pattern}")
    except Exception as e:
        logger.error(f"Ошибка очистки кэша: {e}")

def get_cache_stats() -> Dict:
    """Получение статистики по кэшу"""
    if not redis_client:
        return {"status": "redis_not_available"}
    
    try:
        keys = redis_client.keys("cache:*")
        total_keys = len(keys)
        
        # Подсчет по типам эндпоинтов
        by_type = {}
        for key in keys:
            parts = key.split(":")
            if len(parts) >= 2:
                endpoint_type = parts[1]
                by_type[endpoint_type] = by_type.get(endpoint_type, 0) + 1
        
        return {
            "status": "ok",
            "total_keys": total_keys,
            "by_endpoint_type": by_type
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def validate_geojson_polygon(geojson: dict) -> bool:
    """Проверяет, что GeoJSON содержит валидный полигон"""
    try:
        if geojson.get("type") != "Polygon":
            return False
            
        coordinates = geojson.get("coordinates")
        if not coordinates or not isinstance(coordinates, list):
            return False
            
        for ring in coordinates:
            if len(ring) < 4 or ring[0] != ring[-1]:
                return False
                
        return True
    except:
        return False

def extract_external_id(feature_data: dict) -> Optional[str]:
    """Извлечение external_id из feature_data"""
    if not feature_data or not isinstance(feature_data, dict):
        return None

    meta_info = feature_data.get('meta_info', {})
    if isinstance(meta_info, dict):
        return meta_info.get('id')
    
    return None

def extract_all_external_ids(descriptions: list) -> list:
    """Извлекает все external_id из списка описаний"""
    external_ids = []
    
    for desc in descriptions:
        if isinstance(desc, dict):
            external_id = extract_external_id_from_desc(desc)
            if external_id and external_id not in external_ids:
                external_ids.append(external_id)
    
    return external_ids

def extract_external_id_from_desc(desc_data: dict) -> Optional[str]:
    """Извлечение external_id из данных описания"""
    if not desc_data or not isinstance(desc_data, dict):
        return None
    
    if 'structured_data' in desc_data and isinstance(desc_data['structured_data'], dict):
        structured_data = desc_data['structured_data']
        
        if ('metadata' in structured_data and 
            isinstance(structured_data['metadata'], dict) and
            'meta_info' in structured_data['metadata'] and
            isinstance(structured_data['metadata']['meta_info'], dict)):
            
            meta_info = structured_data['metadata']['meta_info']
            external_id = meta_info.get('id')
            
            if external_id:
                return str(external_id)
    
    return None

def get_proper_title(desc: dict, fallback_name: str = None, index: int = 1) -> str:
    """
    Формирует корректный заголовок в порядке приоритета:
    1. object_name из БД
    2. title из feature_data
    3. Первая строка content (как крайний вариант)
    4. Заголовок по умолчанию
    """
    if not isinstance(desc, dict):
        return f"Описание {index}"
    
    title = desc.get("object_name")
    if title and title.strip():
        return title.strip()
    
    feature_data = desc.get("feature_data", {})
    if isinstance(feature_data, dict):
        title = feature_data.get("title")
        if title and title.strip():
            return title.strip()
    
    structured_data = desc.get("structured_data", {})
    if isinstance(structured_data, dict):
        metadata = structured_data.get("metadata", {})
        if isinstance(metadata, dict):
            meta_info = metadata.get("meta_info", {})
            if isinstance(meta_info, dict):
                title = meta_info.get("title")
                if title and title.strip():
                    return title.strip()
        
        title = structured_data.get("title")
        if title and title.strip():
            return title.strip()
    
    content = desc.get("content", "")
    if content and isinstance(content, str):
        lines = content.strip().split('\n')
        if lines and lines[0].strip():
            first_line = lines[0].strip()
            if len(first_line) > 100:
                return first_line[:97] + "..."
            return first_line
    
    if fallback_name and fallback_name.strip():
        return f"{fallback_name} - описание {index}"
    
    return f"Описание {index}"

def convert_floats(obj):
    """Рекурсивно преобразует numpy float32 и float64 в стандартные float"""
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(item) for item in obj]
    elif hasattr(obj, 'dtype') and ('float32' in str(obj.dtype) or 'float64' in str(obj.dtype)):
        return float(obj)
    elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
        return int(obj)
    return obj