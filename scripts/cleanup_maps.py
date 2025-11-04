# /scripts/cleanup_maps.py

import os
import redis
import logging
from datetime import datetime

# --- НАСТРОЙКИ ---
MAPS_DIR = "/var/www/map_bot/maps"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 1
# Префиксы ключей, которые мы используем в API
REDIS_KEY_PREFIXES = [
    "cache:area_search:",
    "cache:coords_search:", 
    "cache:polygon_simply:"
]
# Расширения файлов карт (проверьте какие реально используются)
FILE_EXTENSIONS = [".jpeg", ".png", ".html"]
# -----------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def cleanup_orphaned_maps():
    """
    Удаляет файлы карт, для которых истек срок действия ключа в Redis.
    """
    logging.info("--- Запуск скрипта очистки старых карт ---")

    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        redis_client.ping()
        logging.info("Успешное подключение к Redis.")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Не удалось подключиться к Redis: {e}")
        return

    if not os.path.isdir(MAPS_DIR):
        logging.error(f"Директория с картами не найдена: {MAPS_DIR}")
        return

    # Получаем все файлы карт
    try:
        map_files = []
        for ext in FILE_EXTENSIONS:
            map_files.extend([f for f in os.listdir(MAPS_DIR) if f.endswith(ext)])
        logging.info(f"Найдено {len(map_files)} файлов карт в директории.")
    except OSError as e:
        logging.error(f"Ошибка чтения директории {MAPS_DIR}: {e}")
        return

    deleted_count = 0
    kept_count = 0
    error_count = 0

    # Проверяем каждый файл
    for filename in map_files:
        try:
            # Из имени файла извлекаем redis_key
            # Формат: map_cache_area_search_xxxxx.jpeg → cache:area_search:xxxxx
            if filename.startswith("map_cache_"):
                # Убираем "map_" префикс и расширение
                clean_name = filename.replace("map_", "").split('.')[0]
                # Заменяем подчеркивания обратно на двоеточия
                redis_key = clean_name.replace("_", ":")
            else:
                # Для старых файлов с другим форматом имен
                clean_name = filename.split('.')[0]
                redis_key = f"cache:map:{clean_name}"

            # Проверяем, существует ли ключ в Redis
            key_exists = False
            for prefix in REDIS_KEY_PREFIXES:
                if redis_key.startswith(prefix):
                    if redis_client.exists(redis_key):
                        key_exists = True
                        break
            
            if not key_exists:
                # Ключа нет - файл "осиротел", удаляем
                file_path = os.path.join(MAPS_DIR, filename)
                try:
                    os.remove(file_path)
                    logging.info(f"Удален осиротевший файл: {filename} (ключ: {redis_key})")
                    deleted_count += 1
                except OSError as e:
                    logging.error(f"Не удалось удалить файл {file_path}: {e}")
                    error_count += 1
            else:
                # Ключ существует - файл актуален
                kept_count += 1
                
        except Exception as e:
            logging.error(f"Ошибка при обработке файла {filename}: {e}")
            error_count += 1

    logging.info("--- Очистка завершена ---")
    logging.info(f"Итог: Проверено - {len(map_files)}, Удалено - {deleted_count}, Оставлено - {kept_count}, Ошибок - {error_count}")

def cleanup_old_redis_keys():
    """
    Дополнительная функция: очистка старых ключей Redis
    которые уже истекли, но файлы могли остаться
    """
    logging.info("--- Очистка старых ключей Redis ---")
    
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        
        deleted_keys = 0
        for prefix in REDIS_KEY_PREFIXES:
            # Ищем все ключи с нашими префиксами
            keys = redis_client.keys(f"{prefix}*")
            for key in keys:
                # Проверяем TTL ключа
                ttl = redis_client.ttl(key)
                if ttl == -2:  # Ключ уже удален
                    continue
                elif ttl == -1:  # Ключ без TTL (должен быть с TTL)
                    logging.warning(f"Найден ключ без TTL: {key}")
                # Можно добавить логику для удаления очень старых ключей
                # если TTL меньше определенного значения
                
        logging.info(f"Проверено ключей Redis: {len(keys)}")
        
    except Exception as e:
        logging.error(f"Ошибка при очистке ключей Redis: {e}")

if __name__ == "__main__":
    cleanup_orphaned_maps()