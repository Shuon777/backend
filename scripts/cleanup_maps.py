# /scripts/cleanup_maps.py

import os
import redis
import logging
from pathlib import Path
import argparse # Для добавления --dry-run

# --- НАСТРОЙКИ ---
# Используем Path для работы с путями - это удобнее и безопаснее
MAPS_DIR = Path("/var/www/map_bot/maps")
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 1
# Префиксы для поиска ключей в Redis
REDIS_KEY_PATTERN = "cache:*:*"
# -----------------

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def get_active_redis_hashes() -> set:
    """Подключается к Redis и возвращает набор всех активных хэшей карт."""
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
        )
        redis_client.ping()
        logging.info("Успешное подключение к Redis.")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Не удалось подключиться к Redis: {e}")
        return set()

    # Получаем ВСЕ ключи, которые соответствуют нашему шаблону
    active_keys = redis_client.keys(REDIS_KEY_PATTERN)
    
    # Из каждого ключа вида "cache:area_search:HASH" извлекаем только HASH
    active_hashes = {key.split(':')[-1] for key in active_keys}
    
    logging.info(f"Найдено {len(active_hashes)} активных ключей/хэшей в Redis DB {REDIS_DB}.")
    return active_hashes

def cleanup_orphaned_maps(dry_run: bool = True):
    """
    Сравнивает файлы на диске с активными хэшами в Redis и удаляет "осиротевшие".
    """
    if dry_run:
        logging.warning("--- Запуск в режиме СУХОГО ПРОГОНА (DRY RUN). Файлы не будут удалены. ---")
    else:
        logging.info("--- Запуск в РАБОЧЕМ режиме. Файлы будут удалены. ---")

    active_hashes = get_active_redis_hashes()
    if not active_hashes:
        logging.warning("Не найдено активных ключей в Redis. Очистка прервана для безопасности.")
        return

    if not MAPS_DIR.is_dir():
        logging.error(f"Директория с картами не найдена: {MAPS_DIR}")
        return

    # Статистика
    files_checked = 0
    files_to_delete = 0
    
    # Итерируемся по всем файлам в директории
    for file_path in MAPS_DIR.glob('*'):
        if file_path.is_file():
            files_checked += 1
            # Из имени файла 'map_area_search_HASH.jpeg' извлекаем 'HASH'
            # rsplit('_', 1) - делит строку по последнему '_'
            # [0] - берем первую часть, [-1] - вторую
            try:
                file_hash = file_path.stem.rsplit('_', 1)[-1]
            except IndexError:
                logging.warning(f"Не удалось извлечь хэш из имени файла: {file_path.name}. Пропускаем.")
                continue

            # Если хэш файла НЕ НАЙДЕН в списке активных хэшей из Redis...
            if file_hash not in active_hashes:
                files_to_delete += 1
                logging.info(f"Найден осиротевший файл: {file_path.name}")
                
                if not dry_run:
                    try:
                        file_path.unlink()
                        logging.warning(f"  -> УДАЛЕН: {file_path.name}")
                    except OSError as e:
                        logging.error(f"  -> ОШИБКА УДАЛЕНИЯ: {e}")

    logging.info("--- Очистка завершена ---")
    if dry_run:
        logging.info(f"Итог (Dry Run): Проверено файлов - {files_checked}. Найдено для удаления - {files_to_delete}.")
        logging.info("Для реального удаления запустите скрипт с флагом --execute")
    else:
        logging.info(f"Итог: Проверено файлов - {files_checked}. Удалено - {files_to_delete}.")

if __name__ == "__main__":
    # Добавляем парсер аргументов для безопасного запуска
    parser = argparse.ArgumentParser(description="Удаляет старые файлы карт, ключи которых истекли в Redis.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Запустить скрипт в рабочем режиме (реально удалять файлы)."
    )
    args = parser.parse_args()

    # По умолчанию dry_run=True (безопасный режим)
    # Если запустить с флагом --execute, то dry_run станет False
    cleanup_orphaned_maps(dry_run=not args.execute)