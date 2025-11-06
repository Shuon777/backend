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
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
        )
        redis_client.ping()
        logging.info("Успешное подключение к Redis.")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Не удалось подключиться к Redis: {e}")
        return

    if not os.path.isdir(MAPS_DIR):
        logging.error(f"Директория с картами не найдена: {MAPS_DIR}")
        return

    try:
        all_files = os.listdir(MAPS_DIR)
        map_files = [
            f for f in all_files 
            if any(f.endswith(ext) for ext in FILE_EXTENSIONS)
        ]
        logging.info(f"Найдено {len(map_files)} файлов карт в директории.")
    except OSError as e:
        logging.error(f"Ошибка чтения директории {MAPS_DIR}: {e}")
        return

    deleted_count = 0
    kept_count = 0
    error_count = 0

    for filename in map_files:
        try:
            # [ТРЕТЬЕ ИСПРАВЛЕНИЕ] Надежная логика на основе rsplit
            
            base_name = os.path.splitext(filename)[0]
            
            key_part = ""
            if base_name.startswith("webapp_map_"):
                key_part = base_name.replace("webapp_map_", "", 1)
            elif base_name.startswith("map_"):
                key_part = base_name.replace("map_", "", 1)
            else:
                logging.warning(f"Неизвестный формат имени файла: {filename}. Пропускаем.")
                error_count += 1
                continue
            
            # Разделяем строку по ПОСЛЕДНЕМУ подчеркиванию
            # "polygon_simply_HASH" -> ["polygon_simply", "HASH"]
            # "area_search_HASH"  -> ["area_search", "HASH"]
            try:
                key_type, key_hash = key_part.rsplit('_', 1)
            except ValueError:
                logging.warning(f"Не удалось разделить на тип и хэш: {key_part}. Пропускаем файл {filename}.")
                error_count += 1
                continue

            # Собираем правильный ключ
            redis_key = f"cache:{key_type}:{key_hash}"

            if not redis_client.exists(redis_key):
                file_path = os.path.join(MAPS_DIR, filename)
                try:
                    os.remove(file_path)
                    logging.info(f"УДАЛЕН осиротевший файл: {filename} (проверялся ключ: {redis_key})")
                    deleted_count += 1
                except OSError as e:
                    logging.error(f"Не удалось удалить файл {file_path}: {e}")
                    error_count += 1
            else:
                logging.info(f"Файл актуален: {filename} (ключ {redis_key} найден)")
                kept_count += 1
                
        except Exception as e:
            logging.error(f"Критическая ошибка при обработке файла {filename}: {e}", exc_info=True)
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