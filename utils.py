# utils.py
import json
import hashlib

def generate_cache_key(params: dict) -> str:
    """
    Создает уникальный MD5 хеш из словаря параметров.
    Ключи сортируются, чтобы порядок не влиял на результат.
    """
    # Преобразуем словарь в каноническую строку JSON
    canonical_string = json.dumps(params, sort_keys=True, ensure_ascii=False).encode('utf-8')
    
    # Возвращаем MD5 хеш этой строки
    return hashlib.md5(canonical_string).hexdigest()