# /scripts/cleanup_maps.py

import os
import redis
import logging
from datetime import datetime

# --- НАСТРОЙКИ ---
# Убедитесь, что эти пути и параметры соответствуют вашему проекту
MAPS_DIR = "/var/www/map_bot/maps"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 1  # Та же база данных, что и в api.py
REDIS_KEY_PREFIX = "cache:map:"
FILE_EXTENSION = ".jpeg"
# -----------------

# Настройка логирования, чтобы видеть, что делает скрипт
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Вывод в консоль
    ]
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
        # Проверяем соединение
        redis_client.ping()
        logging.info("Успешное подключение к Redis.")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Не удалось подключиться к Redis: {e}")
        return

    if not os.path.isdir(MAPS_DIR):
        logging.error(f"Директория с картами не найдена: {MAPS_DIR}")
        return

    # Получаем список всех файлов карт в директории
    try:
        map_files = [f for f in os.listdir(MAPS_DIR) if f.endswith(FILE_EXTENSION)]
        logging.info(f"Найдено {len(map_files)} файлов карт в директории.")
    except OSError as e:
        logging.error(f"Ошибка чтения директории {MAPS_DIR}: {e}")
        return

    deleted_count = 0
    kept_count = 0

    # Проверяем каждый файл
    for filename in map_files:
        # Из имени файла получаем чистый cache_key
        cache_key = filename.removesuffix(FILE_EXTENSION)
        
        # Собираем полный ключ, который должен быть в Redis
        redis_key = f"{REDIS_KEY_PREFIX}{cache_key}"

        try:
            # Проверяем, существует ли еще такой ключ в Redis
            if not redis_client.exists(redis_key):
                # Если ключа нет - файл "осиротел", его нужно удалить
                file_path = os.path.join(MAPS_DIR, filename)
                try:
                    os.remove(file_path)
                    logging.info(f"Удален осиротевший файл: {filename}")
                    deleted_count += 1
                except OSError as e:
                    logging.error(f"Не удалось удалить файл {file_path}: {e}")
            else:
                # Если ключ есть - файл все еще актуален
                kept_count += 1
        except Exception as e:
            logging.error(f"Произошла ошибка при проверке ключа {redis_key}: {e}")

    logging.info("--- Очистка завершена ---")
    logging.info(f"Итог: Проверено файлов - {len(map_files)}, Удалено - {deleted_count}, Оставлено - {kept_count}.")

if __name__ == "__main__":
    cleanup_orphaned_maps()