import json
import logging
import math
import os
import time
from pathlib import Path
from urllib.parse import unquote
import redis
from flask import Flask, jsonify, request
from flask_cors import CORS
from http.client import HTTPException
from shapely.geometry import shape
import shutil  
import hashlib
from core.coordinates_finder import GeoProcessor
from core.relational_service import RelationalService
from core.search_service import SearchService
from embedding_config import embedding_config
from infrastructure.db_utils_for_search import Slot_validator
from infrastructure.geo_db_store import find_place_flexible, get_place
from infrastructure.maps_store import get_map_links
from infrastructure.to_nomn import find_place_key, to_prepositional_phrase
from utils import (
    generate_cache_key, 
    get_cached_result, 
    set_cached_result,
    clear_cache_pattern,
    get_cache_stats,
    init_redis
)
from dotenv import load_dotenv
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import tempfile
import secrets
from functools import wraps
# Добавим импорт нашего сервиса
from core.resource_update_service import ResourceUpdateService
BASE_DIR = Path(__file__).parent
# Константы для путей
RESOURCES_DIST_PATH = str(BASE_DIR / "json_files" / "resources_dist.json")
IMAGES_DIR = "images"

load_dotenv()

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

#Redis settings
REDIS_HOST=os.getenv("REDIS_HOST")
REDIS_PORT=os.getenv("REDIS_PORT")
REDIS_DB=os.getenv("REDIS_DB")

app = Flask(__name__)
CORS(app)

MAPS_DIR = os.getenv("MAPS_DIR")
DOMAIN = os.getenv("PUBLIC_BASE_URL")

geo = GeoProcessor(maps_dir=MAPS_DIR, domain=DOMAIN)
slot_val = Slot_validator()
init_redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

current_dir = Path(__file__).parent
embedding_model_path = str(current_dir / "embedding_models" / "BERTA")

search_service = SearchService(
    embedding_model_path=embedding_model_path
)
relational_service = RelationalService()

user_locations = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "default_admin_password")  
def require_admin_password(f):
    """Декоратор для проверки пароля админа"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_password = request.args.get('admin_password') or request.headers.get('X-Admin-Password')
        
        if not provided_password:
            return jsonify({
                "status": "error",
                "message": "Требуется пароль администратора"
            }), 401
        
        if not secrets.compare_digest(provided_password, ADMIN_PASSWORD):
            return jsonify({
                "status": "error", 
                "message": "Неверный пароль администратора"
            }), 403
            
        return f(*args, **kwargs)
    return decorated_function

@app.route("/log_error", methods=["POST"])
def log_error():
    """
    Логирование ошибок от фронтенда в таблицу error_log через RelationalService
    Формат запроса:
    {
        "user_query": "текст запроса пользователя",  # необязательно
        "error_message": "Описание ошибки",         # обязательно
        "context": {},                              # необязательно, JSON объект с контекстом
        "additional_info": {}                       # необязательно, дополнительная информация
    }
    
    Returns:
        {
            "status": "success" | "error",
            "message": "Сообщение о результате",
            "error_id": 123,  # только при успехе
            "used_objects": [],  # всегда пустой массив
            "not_used_objects": []  # всегда пустой массив
        }
    """
    try:
        data = request.get_json()
        
        # Проверяем обязательные поля
        if not data or "error_message" not in data:
            return jsonify({
                "status": "error",
                "message": "Обязательное поле 'error_message' отсутствует",
                "used_objects": [],
                "not_used_objects": []
            }), 400
        
        # Извлекаем поля
        user_query = data.get("user_query", "")
        error_message = data["error_message"]
        context = data.get("context", {})
        additional_info = data.get("additional_info", {})
        
        logger.info(f"📝 Логирование ошибки: {error_message[:100]}...")
        
        # Используем RelationalService для записи в базу
        success, error_id, message = relational_service.log_error_to_db(
            user_query=user_query,
            error_message=error_message,
            context=context,
            additional_info=additional_info
        )
        
        if success:
            return jsonify({
                "status": "success",
                "message": message,
                "error_id": error_id,
                "used_objects": [],
                "not_used_objects": []
            })
        else:
            logger.error(f"❌ Ошибка при записи в базу данных: {message}")
            
            return jsonify({
                "status": "error",
                "message": message,
                "used_objects": [],
                "not_used_objects": []
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка обработки запроса /log_error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка обработки запроса: {str(e)}",
            "used_objects": [],
            "not_used_objects": []
        }), 500
        
@app.route("/reload_database", methods=["POST"])
@require_admin_password
def reload_database():
    """
    Эндпоинт для перезагрузки базы данных без добавления новых ресурсов
    
    Принимает:
    - reload_database: флаг перезагрузки реляционной базы (true/false) - ОБЯЗАТЕЛЬНЫЙ
    - incremental: флаг инкрементального обновления (true/false)
    - use_stubs: флаг использования заглушек для векторов (true/false)
    """
    try:
        logger.info(f"📤 /reload_database - получен запрос")
        
        reload_database_param = request.form.get('reload_database', 'false').lower() == 'true'
        incremental = request.form.get('incremental', 'true').lower() == 'true'
        use_stubs = request.form.get('use_stubs', 'true').lower() == 'true'
        
        logger.info(f"Параметры запроса:")
        logger.info(f"  - reload_database: {reload_database_param}")
        logger.info(f"  - incremental: {incremental}")
        logger.info(f"  - use_stubs: {use_stubs}")
        
        if not reload_database_param:
            return jsonify({
                "status": "error",
                "message": "Параметр reload_database должен быть true для перезагрузки БД",
                "used_objects": [],
                "not_used_objects": []
            }), 400
        
        # Проверяем существование ресурсного файла
        if not os.path.exists(RESOURCES_DIST_PATH):
            logger.error(f"Файл resources_dist.json не найден: {RESOURCES_DIST_PATH}")
            return jsonify({
                "status": "error",
                "message": f"Файл resources_dist.json не найден",
                "used_objects": [],
                "not_used_objects": []
            }), 404
        
        # Создаем сервис для обработки
        service = ResourceUpdateService(RESOURCES_DIST_PATH, IMAGES_DIR)
        
        # Вызываем метод только для перезагрузки БД
        results = service.reload_database_only(
            reload_database=reload_database_param,
            use_stubs=use_stubs,
            incremental=incremental
        )
        
        # Формируем ответ
        response_data = {
            "status": "success",
            "message": "База данных успешно перезагружена",
            "results": results,
            "used_objects": [
                {
                    "name": "resources_dist.json",
                    "type": "configuration",
                    "operation": "read"
                }
            ],
            "not_used_objects": []
        }
        
        if not results["database_reloaded"]:
            response_data["status"] = "error"
            response_data["message"] = "Ошибка при перезагрузке базы данных"
            if results.get("errors"):
                response_data["message"] += f": {', '.join(results['errors'])}"
        
        logger.info(f"✅ Перезагрузка БД завершена: {results}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Ошибка в /reload_database: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Внутренняя ошибка сервера: {str(e)}",
            "used_objects": [],
            "not_used_objects": []
        }), 500
        
@app.route("/upload_resources", methods=["POST"])
@require_admin_password  
def upload_resources():
    """
    Эндпоинт для загрузки архивов с аннотациями и изображениями
    
    Принимает:
    - json_archive: zip архив с JSON аннотациями (не обязательно)
    - images_archive: zip архив с изображениями (не обязательно)
    - reload_database: флаг перезагрузки реляционной базы (true/false) - ОБЯЗАТЕЛЬНЫЙ для обновления БД
    - incremental_update: флаг инкрементального обновления (true/false)
    - use_stubs: флаг использования заглушек для векторов (true/false)
    """
    try:
        logger.info(f"📤 /upload_resources - получен запрос")
        
        # Проверяем наличие файлов
        has_json = 'json_archive' in request.files and request.files['json_archive'].filename
        has_images = 'images_archive' in request.files and request.files['images_archive'].filename
        
        reload_database = request.form.get('reload_database', 'false').lower() == 'true'
        incremental_update = request.form.get('incremental_update', 'true').lower() == 'true'  # По умолчанию true
        use_stubs = request.form.get('use_stubs', 'true').lower() == 'true'
        
        logger.info(f"Параметры запроса:")
        logger.info(f"  - Есть JSON архив: {has_json}")
        logger.info(f"  - Есть архив изображений: {has_images}")
        logger.info(f"  - reload_database: {reload_database}")
        logger.info(f"  - incremental_update: {incremental_update}")
        logger.info(f"  - use_stubs: {use_stubs}")
        
        # ВАЖНО: Проверяем значения
        logger.info(f"🔍 Проверка значений из request.form:")
        for key in request.form:
            logger.info(f"  - {key}: {request.form.get(key)}")
        
        # Проверяем существование ресурсного файла
        if not os.path.exists(RESOURCES_DIST_PATH):
            logger.error(f"Файл resources_dist.json не найден: {RESOURCES_DIST_PATH}")
            # Создаем файл если не существует
            try:
                os.makedirs(os.path.dirname(RESOURCES_DIST_PATH), exist_ok=True)
                with open(RESOURCES_DIST_PATH, 'w', encoding='utf-8') as f:
                    json.dump({"resources": []}, f, ensure_ascii=False, indent=2)
                logger.info(f"Создан новый файл resources_dist.json")
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": f"Не удалось создать resources_dist.json: {str(e)}",
                    "used_objects": [],
                    "not_used_objects": []
                }), 500
        
        # Создаем временную папку для сохранения архивов
        temp_dir = tempfile.mkdtemp()
        json_archive_path = None
        images_archive_path = None
        
        try:
            # Сохраняем JSON архив если есть
            if has_json:
                json_file = request.files['json_archive']
                filename = secure_filename(json_file.filename)
                json_archive_path = os.path.join(temp_dir, filename)
                json_file.save(json_archive_path)
                logger.info(f"Сохранен JSON архив: {json_archive_path} ({os.path.getsize(json_archive_path)} байт)")
            
            # Сохраняем архив с изображениями если есть
            if has_images:
                images_file = request.files['images_archive']
                filename = secure_filename(images_file.filename)
                images_archive_path = os.path.join(temp_dir, filename)
                images_file.save(images_archive_path)
                logger.info(f"Сохранен архив с изображениями: {images_archive_path} ({os.path.getsize(images_archive_path)} байт)")
            
            # Создаем сервис для обработки
            service = ResourceUpdateService(RESOURCES_DIST_PATH, IMAGES_DIR)
            
            # Обрабатываем загрузку
            results = service.process_upload(
                json_archive_path=json_archive_path,
                images_archive_path=images_archive_path,
                reload_database=reload_database,
                use_stubs=use_stubs,
                incremental=incremental_update  # Передаем параметр
            )
            
            # Добавляем информацию о типе обновления
            if reload_database:
                results["update_type"] = "полное" if not incremental_update else "инкрементальное"
            else:
                results["update_type"] = "без обновления БД"
            
            if reload_database and results.get("database_status") == "started":
                response_data = {
                    "status": "success",
                    "message": "Ресурсы успешно обработаны. Перезагрузка базы данных запущена в фоновом режиме.",
                    "details": results.get("database_message", "Можете следить за прогрессом в логах"),
                    "results": {
                        "json_processed": results.get("json_processed", 0),
                        "images_processed": results.get("images_processed", 0),
                        "new_resources": results.get("new_resources", 0),
                        "updated_resources": results.get("updated_resources", 0),
                        "database_status": "started",
                        "update_type": results.get("update_type", "неизвестно"),
                        "use_stubs": use_stubs
                    },
                    "used_objects": [
                        {
                            "name": "resources_dist.json",
                            "type": "configuration",
                            "operation": "update"
                        }
                    ],
                    "not_used_objects": []
                }
            else:
                # Существующий ответ для случая без перезагрузки БД
                response_data = {
                    "status": "success",
                    "message": "Ресурсы успешно обработаны",
                    "results": results,
                    "used_objects": [
                        {
                            "name": "resources_dist.json",
                            "type": "configuration",
                            "operation": "update"
                        }
                    ],
                    "not_used_objects": []
                }
            
            if results.get("errors"):
                response_data["status"] = "partial_success"
                response_data["message"] = f"Обработка завершена с ошибками"
            
            logger.info(f"✅ Обработка завершена, отправляем немедленный ответ")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки загрузки: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Ошибка обработки: {str(e)}",
                "used_objects": [],
                "not_used_objects": []
            }), 500
            
        finally:
            # Удаляем временную папку
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Удалена временная папка: {temp_dir}")
                except Exception as e:
                    logger.error(f"Ошибка удаления временной папки: {str(e)}")
                
    except Exception as e:
        logger.error(f"❌ Ошибка в /upload_resources: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Внутренняя ошибка сервера: {str(e)}",
            "used_objects": [],
            "not_used_objects": []
        }), 500

# ============================================================================
# НОВЫЙ ЭНДПОИНТ: Загрузка одного изображения с аннотацией
# ============================================================================
@app.route("/upload_single_resource", methods=["POST"])
@require_admin_password
def upload_single_resource():
    """
    Эндпоинт для загрузки одного изображения с JSON аннотацией
    
    Принимает:
    - image_file: файл изображения (jpg, jpeg, png, gif, bmp)
    - json_data: JSON с аннотацией (строка или файл)
    - reload_database: флаг перезагрузки реляционной базы (true/false)
    - incremental_update: флаг инкрементального обновления (true/false)
    - use_stubs: флаг использования заглушек для векторов (true/false)
    - admin_password: пароль администратора (query parameter или заголовок X-Admin-Password)
    
    Формат JSON аннотации должен соответствовать структуре resources_dist.json:
    {
        "featurePhoto2": {
            "name_photo": "путь/к/изображению.jpg",
            "author_photo": "Автор",
            "parent": "Название вида",
            "location": {...},
            "classification_info": {...},
            "date_shooting_time": "...",
            ...
        }
    }
    """
    try:
        logger.info(f"📤 /upload_single_resource - получен запрос")
        
        # Проверяем наличие файла изображения
        if 'image_file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "Необходимо загрузить файл изображения (image_file)",
                "used_objects": [],
                "not_used_objects": []
            }), 400
        
        image_file = request.files['image_file']
        
        # Проверяем, что файл выбран
        if image_file.filename == '':
            return jsonify({
                "status": "error",
                "message": "Файл изображения не выбран",
                "used_objects": [],
                "not_used_objects": []
            }), 400
        
        # Проверяем расширение файла
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}
        file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({
                "status": "error",
                "message": f"Неподдерживаемый формат файла. Допустимы: {', '.join(allowed_extensions)}",
                "used_objects": [],
                "not_used_objects": []
            }), 400
        
        # Получаем JSON аннотацию
        json_data = None
        if 'json_data' in request.files:
            # JSON передан как файл
            json_file = request.files['json_data']
            try:
                json_data = json.load(json_file)
            except json.JSONDecodeError as e:
                return jsonify({
                    "status": "error",
                    "message": f"Ошибка парсинга JSON файла: {str(e)}",
                    "used_objects": [],
                    "not_used_objects": []
                }), 400
        elif 'json_data' in request.form:
            # JSON передан как строка
            try:
                json_data = json.loads(request.form['json_data'])
            except json.JSONDecodeError as e:
                return jsonify({
                    "status": "error",
                    "message": f"Ошибка парсинга JSON строки: {str(e)}",
                    "used_objects": [],
                    "not_used_objects": []
                }), 400
        
        if not json_data:
            return jsonify({
                "status": "error",
                "message": "Необходимо предоставить JSON аннотацию (json_data)",
                "used_objects": [],
                "not_used_objects": []
            }), 400
        
        # Получаем параметры
        reload_database = request.form.get('reload_database', 'false').lower() == 'true'
        incremental_update = request.form.get('incremental_update', 'true').lower() == 'true'
        use_stubs = request.form.get('use_stubs', 'true').lower() == 'true'
        
        # Читаем размер файла для логирования
        image_file.seek(0, 2)  # Переходим в конец файла
        file_size = image_file.tell()
        image_file.seek(0)  # Возвращаемся в начало
        
        logger.info(f"Параметры запроса:")
        logger.info(f"  - Имя файла: {image_file.filename}")
        logger.info(f"  - Размер файла: {file_size} байт")
        logger.info(f"  - Расширение файла: {file_ext}")
        logger.info(f"  - reload_database: {reload_database}")
        logger.info(f"  - incremental_update: {incremental_update}")
        logger.info(f"  - use_stubs: {use_stubs}")
        
        # Создаем временную директорию
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Создана временная директория: {temp_dir}")
        
        try:
            # 1. Сохраняем изображение во временную директорию
            image_filename = secure_filename(image_file.filename)
            image_path = os.path.join(temp_dir, image_filename)
            image_file.save(image_path)
            
            logger.info(f"Изображение сохранено во временную директорию: {image_path}")
            logger.info(f"Файл существует: {os.path.exists(image_path)}")
            logger.info(f"Размер файла: {os.path.getsize(image_path)} байт")
            
            # 2. Создаем временный JSON файл с аннотацией
            json_filename = "annotation.json"
            json_path = os.path.join(temp_dir, json_filename)
            
            # Создаем структуру, аналогичную ресурсам из архивов
            annotation_data = {
                "featurePhoto2": json_data.get("featurePhoto2", json_data)  # Поддерживаем оба формата
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON аннотация сохранена: {json_path}")
            logger.info(f"Структура аннотации: {list(annotation_data.keys())}")
            
            # 3. Проверяем существование ресурсного файла
            if not os.path.exists(RESOURCES_DIST_PATH):
                logger.error(f"Файл resources_dist.json не найден: {RESOURCES_DIST_PATH}")
                # Создаем файл если не существует
                try:
                    os.makedirs(os.path.dirname(RESOURCES_DIST_PATH), exist_ok=True)
                    with open(RESOURCES_DIST_PATH, 'w', encoding='utf-8') as f:
                        json.dump({"resources": []}, f, ensure_ascii=False, indent=2)
                    logger.info(f"Создан новый файл resources_dist.json")
                except Exception as e:
                    logger.error(f"Не удалось создать resources_dist.json: {str(e)}")
                    return jsonify({
                        "status": "error",
                        "message": f"Не удалось создать файл ресурсов: {str(e)}",
                        "used_objects": [],
                        "not_used_objects": []
                    }), 500
            
            # 4. Создаем сервис для обработки
            service = ResourceUpdateService(RESOURCES_DIST_PATH, IMAGES_DIR)
            
            # 5. Обрабатываем как одиночный JSON файл
            logger.info(f"Начинаем обработку JSON файла: {json_path}")
            new_resources_list, new_count, updated_count = service.process_json_file(json_path)
            
            logger.info(f"Результат обработки JSON:")
            logger.info(f"  - Новые ресурсы: {new_count}")
            logger.info(f"  - Обновленные ресурсы: {updated_count}")
            logger.info(f"  - Всего в списке: {len(new_resources_list)}")
            
            if not new_resources_list:
                return jsonify({
                    "status": "error",
                    "message": "Не удалось обработать JSON аннотацию (список ресурсов пуст)",
                    "used_objects": [],
                    "not_used_objects": []
                }), 500
            
            # 6. Копируем изображение в папку images с созданием поддиректорий
            saved_image_path = None
            saved_relative_path = None
            
            try:
                # Определяем путь для изображения на основе JSON данных
                feature_data = annotation_data.get("featurePhoto2", {})
                name_photo = feature_data.get('name_photo', image_filename)
                
                logger.info(f"Путь к изображению из аннотации: '{name_photo}'")
                
                # Извлекаем относительный путь
                relative_path = ""
                
                # Убираем различные префиксы если они есть
                prefixes_to_remove = ['data/images/', 'images/', 'static/images/', 'media/']
                for prefix in prefixes_to_remove:
                    if name_photo.startswith(prefix):
                        relative_path = name_photo[len(prefix):]
                        logger.info(f"Убран префикс '{prefix}': {relative_path}")
                        break
                
                # Если не нашли префикс, используем как есть
                if not relative_path:
                    relative_path = name_photo
                
                # Заменяем пробелы на подчеркивания во всем пути
                relative_path = relative_path.replace(' ', '_')
                logger.info(f"Путь после замены пробелов: '{relative_path}'")
                
                # Получаем имя файла и путь к директории
                filename = os.path.basename(relative_path)
                directory_path = os.path.dirname(relative_path)
                
                logger.info(f"Имя файла: '{filename}'")
                logger.info(f"Путь к директории: '{directory_path}'")
                
                # Создаем полный путь к целевой директории
                target_dir = os.path.join(IMAGES_DIR, directory_path)
                logger.info(f"Полный путь к целевой директории: {target_dir}")
                
                # Создаем все необходимые поддиректории (если их нет)
                os.makedirs(target_dir, exist_ok=True)
                logger.info(f"✅ Создана/проверена директория: {target_dir}")
                
                # Проверяем, что директория создана
                if not os.path.exists(target_dir):
                    logger.error(f"❌ Директория не была создана: {target_dir}")
                    raise FileNotFoundError(f"Не удалось создать директорию: {target_dir}")
                
                # Полный путь к целевому файлу
                target_path = os.path.join(target_dir, filename)
                logger.info(f"Полный путь к целевому файлу: {target_path}")
                
                # Копируем изображение
                shutil.copy2(image_path, target_path)
                logger.info(f"✅ Изображение успешно скопировано")
                
                # Для отладки проверим существование файла
                if os.path.exists(target_path):
                    file_size_kb = os.path.getsize(target_path) / 1024
                    logger.info(f"✅ Файл подтвержден: {target_path}")
                    logger.info(f"✅ Размер файла: {file_size_kb:.2f} KB")
                else:
                    logger.error(f"❌ Файл не найден после копирования: {target_path}")
                    raise FileNotFoundError(f"Файл не был скопирован: {target_path}")
                
                # Сохраняем информацию о пути для ответа
                saved_image_path = target_path
                saved_relative_path = os.path.join(directory_path, filename) if directory_path else filename
                logger.info(f"✅ Сохраненный относительный путь: {saved_relative_path}")
                
            except Exception as e:
                logger.error(f"❌ Ошибка копирования изображения: {str(e)}", exc_info=True)
                return jsonify({
                    "status": "partial_success",
                    "message": f"Аннотация обработана, но не удалось сохранить изображение: {str(e)}",
                    "results": {
                        "json_processed": 1,
                        "images_processed": 0,
                        "new_resources": new_count,
                        "updated_resources": updated_count,
                        "database_reloaded": False
                    },
                    "used_objects": [{"name": image_filename, "type": "image", "operation": "upload"}],
                    "not_used_objects": []
                }), 200
            
            # 7. Перезагружаем базу данных если запрошено
            database_reloaded = False
            if reload_database:
                logger.info(f"🚀 Запрошена перезагрузка БД")
                
                # Создаем временный файл только с новыми ресурсами
                temp_json_file = os.path.join(temp_dir, "new_resources.json")
                with open(temp_json_file, 'w', encoding='utf-8') as f:
                    json.dump({"resources": new_resources_list}, f, ensure_ascii=False, indent=2)
                
                logger.info(f"📄 Создан временный файл с {len(new_resources_list)} ресурсами: {temp_json_file}")
                
                database_reloaded = service.reload_relational_database(
                    reload_database=reload_database,
                    use_stubs=use_stubs,
                    incremental=incremental_update,
                    new_resources_file=temp_json_file
                )
                
                logger.info(f"Результат перезагрузки БД: {database_reloaded}")
            
            # 8. Формируем ответ
            results = {
                "json_processed": 1,
                "images_processed": 1,
                "new_resources": new_count,
                "updated_resources": updated_count,
                "database_reloaded": database_reloaded,
                "update_type": "полное" if not incremental_update else "инкрементальное" if database_reloaded else "без обновления БД",
                "image_saved_path": saved_relative_path,  # Добавляем путь к сохраненному изображению
                "image_full_path": saved_image_path if saved_image_path else None,
                "image_url": f"https://testecobot.ru/images/{saved_relative_path}" if saved_relative_path else None
            }
            
            response_data = {
                "status": "success",
                "message": f"Ресурс успешно обработан и добавлен. Изображение сохранено в: {saved_relative_path}",
                "results": results,
                "used_objects": [
                    {
                        "name": image_filename,
                        "type": "image",
                        "operation": "upload",
                        "saved_path": saved_relative_path,
                        "url": f"https://testecobot.ru/images/{saved_relative_path}" if saved_relative_path else None
                    },
                    {
                        "name": "resources_dist.json",
                        "type": "configuration",
                        "operation": "update"
                    }
                ],
                "not_used_objects": []
            }
            
            # Информация о новом ресурсе
            if new_resources_list:
                resource_info = new_resources_list[0].get('identificator', {})
                resource_id = resource_info.get('id', 'unknown')
                resource_name = resource_info.get('name', {})
                common_name = resource_name.get('common', 'unknown') if isinstance(resource_name, dict) else 'unknown'
                
                response_data["resource_info"] = {
                    "id": resource_id,
                    "name": common_name,
                    "image_path": saved_relative_path,
                    "image_url": f"https://testecobot.ru/images/{saved_relative_path}" if saved_relative_path else None
                }
            
            logger.info(f"✅ Обработка завершена успешно")
            logger.info(f"  - Сохраненное изображение: {saved_relative_path}")
            logger.info(f"  - Новых ресурсов: {new_count}")
            logger.info(f"  - Обновленных ресурсов: {updated_count}")
            logger.info(f"  - БД перезагружена: {database_reloaded}")
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки ресурса: {str(e)}", exc_info=True)
            import traceback
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Ошибка обработки: {str(e)}",
                "used_objects": [],
                "not_used_objects": []
            }), 500
            
        finally:
            # Удаляем временную директорию
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Удалена временная папка: {temp_dir}")
                except Exception as e:
                    logger.error(f"Ошибка удаления временной папки: {str(e)}")
                    
    except Exception as e:
        logger.error(f"❌ Ошибка в /upload_single_resource: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Внутренняя ошибка сервера: {str(e)}",
            "used_objects": [],
            "not_used_objects": []
        }), 500
        
@app.route("/find_off_near_attractions", methods=["POST"])
def find_off_near_attractions():
    """
    Поиск ОФФ около достопримечательностей в указанной области
    Args в JSON body:
    - area_name: название области (например, "Иркутск") - опционально
    - area_polygon: GeoJSON полигона области - опционально (если нет area_name)
    - attraction_types: список типов достопримечательностей (например, ["музей", "памятник"])
    - attraction_subtypes: список подтипов достопримечательностей - опционально
    - off_types: типы ОФФ для поиска (например, ["biological_entity"]) - по умолчанию biological_entity
    - buffer_radius_km: радиус буферной зоны вокруг достопримечательностей в км (по умолчанию 1)
    - limit: ограничение на количество результатов
    - in_stoplist: уровень фильтрации безопасности
    """
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    
    data = request.get_json()
    logger.info(f"🔍 /find_off_near_attractions - запрос: {data}")
    
    # Параметры для кеша
    cache_params = {
        "area_name": data.get("area_name"),
        "area_polygon": data.get("area_polygon"),
        "attraction_types": data.get("attraction_types"),
        "attraction_subtypes": data.get("attraction_subtypes"),
        "off_types": data.get("off_types", ["biological_entity"]),
        "buffer_radius_km": data.get("buffer_radius_km", 1),
        "limit": data.get("limit", 50),
        "in_stoplist": data.get("in_stoplist", "1"),
        "version": "v1"
    }
    
    redis_key = f"cache:off_near_attractions:{generate_cache_key(cache_params)}"
    debug_info = {
        "timestamp": time.time(),
        "cache_key": redis_key,
        "steps": []
    }
    
    # Проверяем кеш
    cache_hit, cached_result = get_cached_result(redis_key, debug_info)
    if cache_hit:
        if debug_mode:
            cached_result["debug"] = debug_info
        return jsonify(cached_result)
    
    # Извлекаем параметры
    area_name = data.get("area_name")
    area_polygon = data.get("area_polygon")
    attraction_types = data.get("attraction_types", [])
    attraction_subtypes = data.get("attraction_subtypes", [])
    off_types = data.get("off_types", ["biological_entity"])
    buffer_radius_km = data.get("buffer_radius_km", 1.0)
    limit = data.get("limit", 50)
    in_stoplist = data.get("in_stoplist", "1")
    
    # Валидация параметров
    if not attraction_types:
        response = {
            "error": "Необходимо указать attraction_types (типы достопримечательностей)",
            "used_objects": [],
            "not_used_objects": []
        }
        return jsonify(response), 400
    
    # Debug информация
    debug_info["parameters"] = {
        "area_name": area_name,
        "has_area_polygon": bool(area_polygon),
        "attraction_types": attraction_types,
        "attraction_subtypes": attraction_subtypes,
        "off_types": off_types,
        "buffer_radius_km": buffer_radius_km,
        "limit": limit,
        "in_stoplist": in_stoplist
    }
    
    try:
        # 1. Получаем полигон области (если указан area_name)
        area_geometry = None
        area_info = None
        
        if area_name:
            # Ищем полигон области
            area_result = relational_service.find_area_geometry(area_name)
            if area_result:
                area_geometry = area_result.get("geometry")
                area_info = area_result.get("area_info", {})
                debug_info["steps"].append({
                    "step": "area_search",
                    "found": True,
                    "area_title": area_info.get('title', area_name)
                })
            else:
                debug_info["steps"].append({
                    "step": "area_search",
                    "found": False,
                    "error": f"Area '{area_name}' not found"
                })
                # Не возвращаем ошибку - продолжаем поиск без ограничения области
        elif area_polygon:
            area_geometry = area_polygon
            debug_info["steps"].append({
                "step": "area_polygon_provided",
                "found": True
            })
        
        # 2. Ищем достопримечательности в области
        attractions = []
        all_attractions = []
        
        for attraction_type in attraction_types:
            search_object_type = "geographical_entity"
            # Ищем достопримечательности
            try:
                if area_geometry:
                    # Поиск в ограниченной области
                    results = relational_service.get_objects_in_area_by_type(
                        area_geometry=area_geometry,
                        object_type=search_object_type,
                        object_subtype=attraction_type if attraction_subtypes else None,
                        limit=limit * 2,  # Ищем больше достопримечательностей
                        search_around=False
                    )
                else:
                    # Поиск по всей БД
                    results = relational_service.search_objects_by_name(
                        object_name="",
                        object_type=search_object_type,
                        object_subtype=attraction_type if attraction_subtypes else None,
                        limit=limit * 2
                    )
                
                if results:
                    # Добавляем тип достопримечательности к объектам
                    for obj in results:
                        obj["attraction_type"] = attraction_type
                    all_attractions.extend(results)
                    
            except Exception as e:
                logger.error(f"Ошибка поиска достопримечательностей типа {attraction_type}: {str(e)}")
                debug_info["steps"].append({
                    "step": f"attraction_search_{attraction_type}",
                    "error": str(e)
                })
        
        # Фильтруем достопримечательности по подтипам если указаны
        if attraction_subtypes:
            filtered_attractions = []
            for obj in all_attractions:
                features = obj.get("features", {})
                geo_type = features.get("geo_type", {})
                specific_types = geo_type.get("specific_types", [])
                
                # Проверяем, есть ли хотя бы один из указанных подтипов
                if any(subtype in specific_types for subtype in attraction_subtypes):
                    filtered_attractions.append(obj)
            attractions = filtered_attractions[:limit]
        else:
            attractions = all_attractions[:limit]
        
        debug_info["attraction_search"] = {
            "total_found": len(all_attractions),
            "after_filtering": len(attractions),
            "attraction_types_used": list(set([a.get("attraction_type", "unknown") for a in attractions]))
        }
        
        if not attractions:
            response = {
                "status": "no_attractions",
                "message": f"В указанной области не найдено достопримечательностей типов: {attraction_types}",
                "used_objects": [],
                "not_used_objects": []
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response)
        
        # 3. Ищем ОФФ около каждой достопримечательности
        all_off_results = []
        attraction_off_map = {}  # Карта достопримечательность -> список ОФФ
        
        for attraction in attractions:
            attraction_id = attraction.get("id")
            attraction_name = attraction.get("name", "Неизвестная достопримечательность")
            attraction_geojson = attraction.get("geojson")
            
            if not attraction_geojson:
                continue
            
            # Для каждой достопримечательности создаем буферную зону
            try:
                # Создаем буферную зону
                buffer_geometry = search_service.geo_service.create_buffer_geometry(
                    attraction_geojson, 
                    buffer_radius_km
                )
                
                if not buffer_geometry:
                    continue
                
                # Ищем ОФФ в буферной зоне
                for off_type in off_types:
                    off_results = search_service.geo_service.get_objects_in_polygon(
                        polygon_geojson=buffer_geometry,
                        object_type=off_type,
                        limit=20  # Лимит на ОФФ для одной достопримечательности
                    )
                    
                    if off_results:
                        # Обрезаем полигоны ОФФ по буферной зоне (если это полигоны)
                        clipped_results = []
                        for off_obj in off_results:
                            # Сохраняем оригинальную геометрию
                            original_geojson = off_obj.get("geojson")
                            
                            # Пытаемся обрезать если это полигон
                            try:
                                if original_geojson and original_geojson.get("type") in ["Polygon", "MultiPolygon"]:
                                    # Используем shapely для обрезки
                                    from shapely.geometry import shape, mapping
                                    from shapely.ops import unary_union
                                    
                                    buffer_shape = shape(buffer_geometry)
                                    off_shape = shape(original_geojson)
                                    
                                    if buffer_shape.intersects(off_shape):
                                        intersection = buffer_shape.intersection(off_shape)
                                        if not intersection.is_empty:
                                            clipped_geojson = mapping(intersection)
                                            off_obj["geojson"] = clipped_geojson
                                        else:
                                            # Если пересечение пустое, пропускаем
                                            continue
                                    else:
                                        # Нет пересечения
                                        continue
                                # Для точек просто проверяем находится ли внутри буфера
                                elif original_geojson and original_geojson.get("type") == "Point":
                                    buffer_shape = shape(buffer_geometry)
                                    off_shape = shape(original_geojson)
                                    if not buffer_shape.contains(off_shape):
                                        continue
                                    
                            except Exception as e:
                                logger.warning(f"Ошибка обрезки геометрии ОФФ: {str(e)}")
                                # Продолжаем с оригинальной геометрией
                            
                            # Добавляем информацию о достопримечательности
                            off_obj["near_attraction"] = {
                                "id": attraction_id,
                                "name": attraction_name,
                                "type": attraction.get("attraction_type", "unknown"),
                                "distance_km": buffer_radius_km
                            }
                            
                            clipped_results.append(off_obj)
                        
                        if clipped_results:
                            # Убираем дубликаты ОФФ (по id или name+coordinates)
                            unique_off = {}
                            for off_obj in clipped_results:
                                key = off_obj.get("id") or f"{off_obj.get('name')}_{off_obj.get('geojson', {}).get('coordinates', [])}"
                                if key not in unique_off:
                                    unique_off[key] = off_obj
                            
                            attraction_off_map.setdefault(attraction_id, []).extend(list(unique_off.values()))
                            all_off_results.extend(list(unique_off.values()))
                            
            except Exception as e:
                logger.error(f"Ошибка поиска ОФФ около {attraction_name}: {str(e)}")
                debug_info["steps"].append({
                    "step": f"off_search_near_{attraction_id}",
                    "error": str(e)
                })
        
        # 4. Подготавливаем данные для карты
        map_objects = []
        used_objects = []
        not_used_objects = []
        
        # Добавляем область если есть
        if area_geometry:
            map_objects.append({
                'tooltip': f"Область поиска: {area_info.get('title', area_name) if area_name else 'Пользовательский полигон'}",
                'popup': f"<h6>Область поиска</h6><p>Поиск достопримечательностей в этой области</p>",
                'geojson': area_geometry,
                'style': {'color': 'blue', 'fillOpacity': 0.1, 'weight': 2}
            })
        
        # Добавляем достопримечательности
        for attraction in attractions:
            attraction_id = attraction.get("id")
            attraction_name = attraction.get("name")
            attraction_type = attraction.get("attraction_type", "unknown")
            geojson = attraction.get("geojson")
            
            used_objects.append({
                "name": attraction_name,
                "type": "attraction",
                "attraction_type": attraction_type,
                "id": attraction_id
            })
            
            # Создаем popup с информацией о найденных ОФФ
            off_list = attraction_off_map.get(attraction_id, [])
            off_count = len(off_list)
            
            popup_html = f"<h6>{attraction_name}</h6>"
            popup_html += f"<p>Тип: {attraction_type}</p>"
            
            if off_count > 0:
                popup_html += f"<p>Найдено ОФФ поблизости: {off_count}</p>"
                popup_html += "<ul>"
                for off in off_list[:5]:  # Показываем первые 5
                    popup_html += f"<li>{off.get('name', 'Неизвестный ОФФ')}</li>"
                if off_count > 5:
                    popup_html += f"<li>... и еще {off_count - 5}</li>"
                popup_html += "</ul>"
            else:
                popup_html += "<p>ОФФ поблизости не найдены</p>"
            
            map_objects.append({
                'tooltip': f"{attraction_name} ({attraction_type})",
                'popup': popup_html,
                'geojson': geojson,
                'style': {'color': 'red', 'fillOpacity': 0.3, 'weight': 3}
            })
            
            # Добавляем буферную зону
            try:
                buffer_geometry = search_service.geo_service.create_buffer_geometry(
                    geojson, 
                    buffer_radius_km
                )
                if buffer_geometry:
                    map_objects.append({
                        'tooltip': f"Зона поиска ОФФ вокруг {attraction_name}",
                        'popup': f"<h6>Буферная зона</h6><p>Зона поиска ОФФ в радиусе {buffer_radius_km} км вокруг {attraction_name}</p>",
                        'geojson': buffer_geometry,
                        'style': {'color': 'orange', 'fillOpacity': 0.1, 'weight': 2}
                    })
            except Exception as e:
                logger.warning(f"Не удалось создать буфер для {attraction_name}: {str(e)}")
        
        # Добавляем ОФФ
        off_names = set()
        for off in all_off_results:
            off_name = off.get("name", "Неизвестный ОФФ")
            off_type = off.get("type", "unknown")
            geojson = off.get("geojson")
            near_attraction = off.get("near_attraction", {})
            
            if off_name not in off_names:
                off_names.add(off_name)
                used_objects.append({
                    "name": off_name,
                    "type": off_type,
                    "near_attraction": near_attraction.get("name")
                })
            
            popup_html = f"<h6>{off_name}</h6>"
            popup_html += f"<p>Тип: {off_type}</p>"
            if near_attraction:
                popup_html += f"<p>Найден около: {near_attraction.get('name')} ({near_attraction.get('type')})</p>"
                popup_html += f"<p>Расстояние: до {near_attraction.get('distance_km')} км</p>"
            
            map_objects.append({
                'tooltip': off_name,
                'popup': popup_html,
                'geojson': geojson,
                'style': {'color': 'green', 'fillOpacity': 0.5, 'weight': 2}
            })
        
        # 5. Создаем карту
        if not map_objects:
            response = {
                "status": "no_results",
                "message": "Не найдено объектов для отображения на карте",
                "used_objects": used_objects,
                "not_used_objects": not_used_objects
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response)
        
        try:
            map_name = redis_key.replace("cache:", "map_").replace(":", "_")
            map_result = geo.draw_custom_geometries(map_objects, map_name)
            
            # Формируем итоговый ответ
            response_data = {
                "map": map_result,
                "statistics": {
                    "attractions_found": len(attractions),
                    "off_found": len(all_off_results),
                    "unique_off_names": len(off_names),
                    "attraction_types": list(set([a.get("attraction_type", "unknown") for a in attractions])),
                    "buffer_radius_km": buffer_radius_km
                },
                "used_objects": used_objects,
                "not_used_objects": not_used_objects,
                "in_stoplist_filter_applied": True,
                "in_stoplist_level": in_stoplist
            }
            
            # Формируем текстовый ответ
            if area_name:
                response_data["answer"] = f"В области '{area_name}' найдено {len(attractions)} достопримечательностей. Около них обнаружено {len(all_off_results)} ОФФ ({len(off_names)} уникальных видов)."
            else:
                response_data["answer"] = f"Найдено {len(attractions)} достопримечательностей. Около них обнаружено {len(all_off_results)} ОФФ ({len(off_names)} уникальных видов)."
            
            # Добавляем списки объектов
            response_data["attractions"] = [
                {
                    "id": a.get("id"),
                    "name": a.get("name"),
                    "type": a.get("attraction_type"),
                    "off_count": len(attraction_off_map.get(a.get("id"), []))
                }
                for a in attractions[:10]  # Ограничиваем для ответа
            ]
            
            response_data["off_objects"] = [
                {
                    "name": off.get("name"),
                    "type": off.get("type"),
                    "near_attraction": off.get("near_attraction", {}).get("name")
                }
                for off in all_off_results[:20]  # Ограничиваем для ответа
            ]
            
            # Debug информация
            if debug_mode:
                debug_info["results_summary"] = {
                    "total_map_objects": len(map_objects),
                    "attraction_off_distribution": {aid: len(offs) for aid, offs in attraction_off_map.items()}
                }
                response_data["debug"] = debug_info
            
            # Сохраняем в кеш
            set_cached_result(redis_key, response_data, expire_time=3600)
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Ошибка создания карты: {str(e)}")
            error_response = {
                "error": f"Ошибка создания карты: {str(e)}",
                "statistics": {
                    "attractions_found": len(attractions),
                    "off_found": len(all_off_results)
                },
                "used_objects": used_objects,
                "not_used_objects": not_used_objects
            }
            if debug_mode:
                error_response["debug"] = debug_info
            return jsonify(error_response), 500
            
    except Exception as e:
        logger.error(f"Ошибка в /find_off_near_attractions: {str(e)}", exc_info=True)
        error_response = {
            "error": f"Внутренняя ошибка сервера: {str(e)}",
            "used_objects": [],
            "not_used_objects": []
        }
        if debug_mode:
            error_response["debug"] = debug_info
        return jsonify(error_response), 500
           
@app.route("/objects_in_polygon_simply", methods=["POST"])
def objects_in_polygon_simply():
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    in_stoplist = request.args.get("in_stoplist", "1")
    logger.info(f"📦 /objects_in_polygon_simply - GET params: {dict(request.args)}")
    logger.info(f"📦 /objects_in_polygon_simply - POST data: {request.get_json()}")

    data = request.get_json()
    name = data.get("name")
    buffer_radius_km = data.get("buffer_radius_km", 0)
    object_type = data.get("object_type")
    object_subtype = data.get("object_subtype")
    limit = data.get("limit", 1500)

    # Параметры для кеша основного запроса
    cache_params = {
        "name": name,
        "buffer_radius_km": buffer_radius_km,
        "object_type": object_type,
        "object_subtype": object_subtype,
        "limit": limit,
        "in_stoplist": in_stoplist,
        "version": "v2"
    }
    redis_key = f"cache:polygon_simply:{generate_cache_key(cache_params)}"
    debug_info = {
        "timestamp": time.time(),
        "cache_key": redis_key,
        "steps": []
    }
    
    cache_hit, cached_result = get_cached_result(redis_key, debug_info)
    if cache_hit:
        if debug_mode:
            cached_result["debug"] = debug_info
        return jsonify(cached_result)

    # Debug информация о параметрах
    debug_info["parameters"] = {
        "name": name,
        "buffer_radius_km": buffer_radius_km,
        "object_type": object_type,
        "object_subtype": object_subtype,
        "limit": limit,
        "in_stoplist": in_stoplist
    }
    
    # Проверяем синонимы перед поиском геометрии
    try:
        resolved_info = search_service.resolve_object_synonym(name, "all")

        if resolved_info.get("resolved", False):
            canonical_name = resolved_info["main_form"]
            logger.debug(f"Найден синоним: '{name}' -> '{canonical_name}' (тип: {resolved_info['object_type']})")
            name = canonical_name
            debug_info["steps"].append({
                "step": "universal_synonym_resolution", 
                "original_name": data.get("name"),
                "canonical_name": canonical_name,
                "object_type": resolved_info["object_type"],
                "resolved_info": resolved_info
            })
    except Exception as e:
        logger.warning(f"Ошибка при проверке синонимов для '{name}': {e}")
        debug_info["steps"].append({
            "step": "synonym_resolution",
            "error": str(e)
        })
    
    entry = relational_service.find_geometry(name)
    if not entry or "geometry" not in entry:
        flexible_result = find_place_flexible(name)
        if flexible_result and flexible_result.get("status") == "found":
            entry = flexible_result["record"]
            logger.debug(f"Найдено через гибкий поиск: '{name}' -> '{flexible_result['name']}'")
            debug_info["steps"].append({
                "step": "flexible_search",
                "found_name": flexible_result['name'],
                "original_name": name
            })
        else:
            logger.debug(f"Геометрия для '{name}' не найдена")
            response = {"error": f"Геометрия для '{name}' не найдена"}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 404
    
    polygon = entry["geometry"]
    debug_info["geometry_source"] = {
        "source": "database" if entry else "flexible_search",
        "entry_id": entry.get("id", "unknown") if entry else "unknown"
    }

    if not polygon:
        response = {"error": "Polygon not specified"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400
    
    try:
        # Получаем объекты из search_service
        results = search_service.get_objects_in_polygon(
            polygon_geojson=polygon,
            buffer_radius_km=float(buffer_radius_km),
            object_type=object_type,
            object_subtype=object_subtype,
            limit=int(limit)
        )

        objects = results.get("objects", [])
        answer = results.get("answer", "")
        
        # [СЧЕТЧИК 1] Всего найдено в БД (сырые данные)
        count_raw_from_db = len(objects)
        
        # ДИАГНОСТИКА: Собираем статистику ДО фильтрации
        total_objects_before = len(objects)
        biological_objects_before = [obj for obj in objects if obj.get('type') in ['Объект флоры','Объект фауны']]
        biological_names_before = list(set(obj.get('name', 'Без имени') for obj in biological_objects_before))
        
        debug_info["before_filter"] = {
            "total_objects": total_objects_before,
            "biological_objects_count": len(biological_objects_before),
            "biological_names": biological_names_before
        }
        
        safe_objects = []
        stoplisted_objects = []
        
        for obj in objects:
            feature_data = obj.get("features", {})
            obj_in_stoplist = feature_data.get("in_stoplist")
            
            try:
                requested_level = int(in_stoplist)
                if obj_in_stoplist is None or int(obj_in_stoplist) <= requested_level:
                    safe_objects.append(obj)
                else:
                    stoplisted_objects.append(obj)
                    # logger.info(f"Исключен объект с in_stoplist={obj_in_stoplist}: {obj.get('name', 'Без имени')}")
            except (ValueError, TypeError):
                if obj_in_stoplist is None or int(obj_in_stoplist) <= 1:
                    safe_objects.append(obj)
                else:
                    stoplisted_objects.append(obj)
                    # logger.info(f"Исключен объект с in_stoplist={obj_in_stoplist}: {obj.get('name', 'Без имени')}")
        
        objects = safe_objects
        
        # [СЧЕТЧИК 2] Осталось после фильтра безопасности
        count_safe_after_filter = len(objects)

        # ДИАГНОСТИКА: Собираем статистику ПОСЛЕ фильтрации
        total_objects_after = len(objects)
        biological_objects_after = [obj for obj in objects if obj.get('type') in ['Объект флоры','Объект фауны']]
        biological_names_after = list(set(obj.get('name', 'Без имени') for obj in biological_objects_after))
        
        # ВАЖНОЕ ИСПРАВЛЕНИЕ: Формируем all_biological_names ПОСЛЕ фильтрации
        all_biological_names = sorted(biological_names_after)
        
        # Обновляем ответ с учетом фильтрации
        if stoplisted_objects:
            answer = f"{answer} (исключено {len(stoplisted_objects)} объектов по уровню безопасности)"
        
        # Debug информация о результатах поиска и фильтрации
        debug_info["search_results"] = {
            "total_objects": len(objects),
            "object_types": {},
            "polygon_area": "calculated" if polygon else "unknown"
        }
        
        debug_info["stoplist_filter"] = {
            "total_before_filter": total_objects_before,
            "safe_after_filter": total_objects_after,
            "stoplisted_count": len(stoplisted_objects),
            "biological_before_filter": len(biological_objects_before),
            "biological_after_filter": len(biological_objects_after),
            "biological_names_before": biological_names_before,
            "biological_names_after": biological_names_after
        }
        
        # Статистика по типам объектов
        for obj in objects:
            obj_type = obj.get("type", "unknown")
            if obj_type not in debug_info["search_results"]["object_types"]:
                debug_info["search_results"]["object_types"][obj_type] = 0
            debug_info["search_results"]["object_types"][obj_type] += 1
            
    except ValueError:
        response = {"error": "Invalid parameters format"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400
    except Exception as e:
        logger.error(f"Ошибка при поиске объектов в полигоне: {e}")
        debug_info["search_error"] = str(e)
        response = {"error": "Внутренняя ошибка сервера при поиске"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 500

    if not objects:
        response = {
            "status": "no_objects", 
            "message": answer,
            "used_objects": [],
            "not_used_objects": [],
            "all_biological_names": []  # Явно указываем пустой массив
        }
        if debug_mode:
            response["debug"] = debug_info
            response["in_stoplist_filter_applied"] = True
            response["in_stoplist_level"] = in_stoplist
        return jsonify(response)

    # Группируем объекты по геометрии и типу
    grouped_by_geojson = {}
    
    # [СЧЕТЧИКИ ДИАГНОСТИКИ ГРУППИРОВКИ]
    count_missing_geo = 0       # Нет координат в JSON
    count_duplicates_in_point = 0 # Одинаковое имя в одной точке (схлопнуто)
    count_total_in_popups = 0   # Реально попало в списки попапов (уникальные имена в точках)
    
    for obj in objects:
        if 'geojson' not in obj or not obj['geojson']:
            count_missing_geo += 1
            continue
        geojson_key = json.dumps(obj['geojson'], sort_keys=True)
        obj_type = obj.get('type', 'unknown')
        
        # ИСПРАВЛЕНИЕ: Сохраняем название географического объекта
        location_name = obj.get('location_name') or obj.get('name') or 'Неизвестное место'
        
        if geojson_key not in grouped_by_geojson:
            grouped_by_geojson[geojson_key] = {
                'geojson': obj['geojson'],
                'type': obj_type,
                'location_name': location_name,  # Сохраняем название геообъекта
                'biological_names': []           # Переименовываем для ясности
            }
        
        object_name = obj.get('name', 'Без имени')
        
        # Логика подсчета того, что попадет в попап
        if object_name not in grouped_by_geojson[geojson_key]['biological_names']:
            grouped_by_geojson[geojson_key]['biological_names'].append(object_name)
            count_total_in_popups += 1
        else:
            count_duplicates_in_point += 1

    # [ЛОГИРОВАНИЕ ИТОГОВОЙ СТАТИСТИКИ]
    logger.debug(
        f"📊 STATISTICS: "
        f"RawDB={count_raw_from_db} | "
        f"Filtered={count_safe_after_filter} | "
        f"NoGeo={count_missing_geo} | "
        f"Dupes(Hidden)={count_duplicates_in_point} | "
        f"VisibleInPopups={count_total_in_popups} | "
        f"UniqueMapMarkers={len(grouped_by_geojson)}"
    )
    logger.debug("📍 MARKER CONTENT BREAKDOWN:")
    for geo_key, group in grouped_by_geojson.items():
        count = len(group['biological_names'])
        # Определяем тип геометрии (Point или Polygon)
        g_type = group['geojson'].get('type', 'Unknown')
        name = group.get('location_name', 'Unknown')
        # Выводим только если там много объектов, чтобы не спамить
        if count > 10:
            logger.debug(f"   🚩 {g_type} at '{name}': contains {count} items")
        else:
            # Для мелких можно кратко
            pass # или logger.debug(f"   🔹 {g_type} at '{name}': {count}")
        
    # Формируем объекты для карты с красивым скроллбаром
    objects_for_map = []
    MAX_NAMES_IN_TOOLTIP = 3
    MAX_VISIBLE_ITEMS = 8  # Показывать скролл после 8 элементов
    MAX_POPUP_HEIGHT = "300px"  # Комфортная высота для большинства экранов

    # Собираем информацию об объектах для фронтенда
    used_objects = []
    not_used_objects = []

    for group_data in grouped_by_geojson.values():
        biological_names = sorted(group_data['biological_names'])
        location_name = group_data.get('location_name', 'Неизвестное место')
        obj_type = group_data.get('type', 'unknown')
        
        # Добавляем объекты в used_objects с названием геообъекта
        used_objects.append({
            "name": location_name,
            "type": obj_type
        })
        
        # Создаем краткий текст для Tooltip
        if len(biological_names) > MAX_NAMES_IN_TOOLTIP:
            tooltip_text = f"{location_name}: {len(biological_names)} видов"
        else:
            tooltip_text = f"{location_name}: {', '.join(biological_names)}"

        # Создаем красивый HTML для Popup с кастомным скроллбаром
        popup_html = f"""
        <div style="max-width: 320px; font-family: Arial, sans-serif;">
            <h5 style="
                margin: 0 0 12px 0; 
                padding: 0; 
                color: #2c3e50; 
                border-bottom: 2px solid #3498db; 
                padding-bottom: 8px;
                font-size: 16px;
            ">{location_name}</h5>
        """

        if obj_type == "biological_entity":
            popup_html += f'''
            <div style="
                font-size: 13px; 
                color: #7f8c8d; 
                margin-bottom: 12px;
                padding: 5px;
                background: #ecf0f1;
                border-radius: 4px;
            ">
                🐾 Обнаружено видов: <strong>{len(biological_names)}</strong>
            </div>
            '''
        else:
            popup_html += f'''
            <div style="
                font-size: 13px; 
                color: #7f8c8d; 
                margin-bottom: 12px;
                padding: 5px;
                background: #ecf0f1;
                border-radius: 4px;
            ">
                📍 Обнаружено объектов: <strong>{len(biological_names)}</strong>
            </div>
            '''

        # Контейнер списка с кастомным скроллбаром
        popup_html += f'''
        <div style="
            max-height: {MAX_POPUP_HEIGHT};
            overflow-y: auto;
            border: 1px solid #bdc3c7;
            border-radius: 6px;
            padding: 8px;
            background: #f8f9fa;
        ">
            <ul style="
                list-style: none;
                padding: 0;
                margin: 0;
            ">
        '''

        for i, biological_name in enumerate(biological_names):
            # Чередование цветов фона для лучшей читаемости
            bg_color = "#ffffff" if i % 2 == 0 else "#f8f9fa"
            
            popup_html += f'''
            <li style="
                padding: 8px 10px;
                margin: 3px 0;
                background: {bg_color};
                border-left: 4px solid #3498db;
                border-radius: 4px;
                font-size: 13px;
                transition: all 0.2s ease;
            ">{biological_name}</li>
            '''

        popup_html += "</ul></div>"

        # Стили для кастомного скроллбара
        popup_html += """
        <style>
            div::-webkit-scrollbar {
                width: 8px;
            }
            div::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 4px;
            }
            div::-webkit-scrollbar-thumb {
                background: #c1c1c1;
                border-radius: 4px;
            }
            div::-webkit-scrollbar-thumb:hover {
                background: #a8a8a8;
            }
            li:hover {
                background: #e3f2fd !important;
                transform: translateX(2px);
            }
        </style>
        </div>
        """
        
        objects_for_map.append({
            'tooltip': tooltip_text,
            'popup': popup_html,
            'geojson': group_data['geojson']
        })

    # =========================================================================
    # [FIX] СОРТИРОВКА ПО ПЛОЩАДИ: Большие полигоны вниз, маленькие и точки наверх
    # =========================================================================
    try:
        def get_sort_key(item):
            """
            Возвращает площадь для сортировки. 
            Точки и Линии (площадь 0) получают приоритет -1, чтобы быть в конце списка (наверху).
            Полигоны сортируются по убыванию площади (сначала большие, потом маленькие).
            """
            try:
                geom = shape(item['geojson'])
                area = geom.area
                # Если площадь 0 (точка или линия) или очень маленькая
                if area == 0 or geom.geom_type in ['Point', 'MultiPoint', 'LineString', 'MultiLineString']:
                    return -1 
                return area
            except:
                return 0

        # Сортируем: 
        # reverse=True -> Сначала большие числа (большая площадь), в конце маленькие/-1.
        # Порядок отрисовки: Элемент[0] (Дно) -> ... -> Элемент[N] (Верх)
        objects_for_map.sort(key=get_sort_key, reverse=True)
        
        logger.debug(f"📐 Objects sorted by area for correct Z-indexing. Count: {len(objects_for_map)}")
    except Exception as e:
        logger.warning(f"Ошибка при сортировке геометрий: {e}")
    # =========================================================================

    try:
        # Создаем карту
        map_name = redis_key.replace("cache:", "map_").replace(":", "_")
        map_result = geo.draw_custom_geometries(objects_for_map, map_name)
        
        # ВАЖНО: Возвращаем число объектов ПОСЛЕ фильтрации безопасности, 
        # о которых мы говорим пользователю (даже если они визуально схлопнулись)
        map_result["count"] = count_safe_after_filter
        
        map_result["answer"] = answer
        map_result["grouped_names"] = [obj.get("tooltip", "") for obj in objects_for_map]
        
        # ВАЖНОЕ ИСПРАВЛЕНИЕ: Используем all_biological_names, сформированный ПОСЛЕ фильтрации
        map_result["all_biological_names"] = all_biological_names
        
        # Добавляем информацию об объектах для фронтенда
        map_result["used_objects"] = used_objects
        map_result["not_used_objects"] = not_used_objects
        
        # Добавляем информацию о фильтрации по stoplist
        map_result["in_stoplist_filter_applied"] = True
        map_result["in_stoplist_level"] = in_stoplist
        map_result["stoplisted_count"] = len(stoplisted_objects)
        
        # Добавляем debug информацию
        if debug_mode:
            debug_info["visualization"] = {
                "map_name": map_name,
                "objects_count": len(objects_for_map),
                "biological_names_count": len(all_biological_names),
                "biological_names_list": all_biological_names,  # Для отладки
                "popup_style": "custom_scrollbar_v2",
                # Добавляем нашу статистику в debug ответ
                "stats_counters": {
                    "raw_db": count_raw_from_db,
                    "safe_filtered": count_safe_after_filter,
                    "no_geo": count_missing_geo,
                    "duplicates_in_point": count_duplicates_in_point,
                    "visible_in_popups": count_total_in_popups
                }
            }
            map_result["debug"] = debug_info

        # Сохраняем в кеш
        set_cached_result(redis_key, map_result, expire_time=2700)
        
        return jsonify(map_result)
        
    except Exception as e:
        logger.error(f"Ошибка отрисовки карты: {e}", exc_info=True)
        debug_info["visualization_error"] = str(e)
        response = {
            "status": "error", 
            "message": f"Ошибка отрисовки карты: {e}",
            "used_objects": [],
            "not_used_objects": [],
            "all_biological_names": all_biological_names  # Сохраняем даже при ошибке
        }
        if debug_mode:
            response["debug"] = debug_info
            response["in_stoplist_filter_applied"] = True
            response["in_stoplist_level"] = in_stoplist
        return jsonify(response), 500
     
@app.route("/objects_in_area_by_type", methods=["POST"])
def objects_in_area_by_type():
    data = request.get_json()
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"

    # Параметры для кеша
    cache_params = {
        "area_name": data.get("area_name"),
        "object_type": data.get("object_type", "all"),
        "object_subtype": data.get("object_subtype"),
        "object_name": data.get("object_name"),
        "limit": data.get("limit", 20),
        "search_around": data.get("search_around", False),
        "buffer_radius_km": data.get("buffer_radius_km", 10.0),
        "version": "v2"
    }
    
    redis_key = f"cache:area_search:{generate_cache_key(cache_params)}"
    debug_info = {
        "timestamp": time.time(),
        "cache_key": redis_key,
        "steps": []
    }

    # Проверяем кеш
    cache_hit, cached_result = get_cached_result(redis_key, debug_info)
    if cache_hit:
        if debug_mode:
            cached_result["debug"] = debug_info
        return jsonify(cached_result)

    logger.info(f"📦 /objects_in_area_by_type - GET params: {dict(request.args)}")
    logger.info(f"📦 /objects_in_area_by_type - POST data: {request.get_json()}")

    area_name = data.get("area_name")
    object_type = data.get("object_type", "all") 
    object_subtype = data.get("object_subtype")
    object_name = data.get("object_name")
    limit = data.get("limit", 1500)
    search_around = data.get("search_around", False)
    buffer_radius_km = data.get("buffer_radius_km", 10.0)

    debug_info["parameters"] = {
        "area_name": area_name,
        "object_type": object_type,
        "object_subtype": object_subtype,
        "object_name": object_name,
        "limit": limit,
        "search_around": search_around,
        "buffer_radius_km": buffer_radius_km
    }
    resolved_object_info = None
    if object_name:
        resolved_object_info = search_service.resolve_object_synonym(object_name, object_type)
        
        debug_info["synonym_resolution"] = {
            "original_name": object_name,
            "original_type": object_type,
            "resolved_info": resolved_object_info
        }
        
        if resolved_object_info.get("resolved", False):
            object_name = resolved_object_info["main_form"]
            if object_type != "all":
                object_type = resolved_object_info["object_type"]
            logger.info(f"✅ Разрешен синоним объекта: '{resolved_object_info['original_name']}' -> '{object_name}' (тип: {object_type})")
        else:
            logger.info(f"ℹ️ Синоним для объекта '{object_name}' не найден, используем оригинальное название")
    
    def extract_external_id(feature_data):
        """Упрощенная функция извлечения ID из feature_data"""
        if not feature_data or not isinstance(feature_data, dict):
            return None

        meta_info = feature_data.get('meta_info', {})
        if isinstance(meta_info, dict):
            return meta_info.get('id')
        
        return None
    
    if not area_name and object_name:
        debug_info["steps"].append({
            "step": "direct_object_search",
            "reason": "area_name not provided, searching object directly",
            "resolved_name": object_name,
            "resolved_type": object_type
        })
        
        try:
            results = search_service.search_objects_directly_by_name(
                object_name=object_name,
                object_type=object_type,
                object_subtype=object_subtype,
                limit=limit
            )
            
            objects = results.get("objects", [])
            answer = results.get("answer", "")
            
            debug_info["search_results"] = {
                "total_objects_found": len(objects),
                "search_type": "direct_object_search"
            }
            
            if not objects:
                response = {
                    "status": "no_objects", 
                    "message": answer
                }
                if debug_mode:
                    response["debug"] = debug_info
                return jsonify(response)
            
            objects_for_map = []
            used_objects = []
            
            for obj in objects:
                name = obj.get('name', 'Без имени')
                description = obj.get('description', '')
                geojson = obj.get('geojson', {})
                obj_type = obj.get('type', 'unknown')
                
                # Добавляем объект в used_objects
                used_objects.append({
                    "name": name,
                    "type": obj_type,
                    "external_id": extract_external_id(obj.get('features', {})),
                    "geometry_type": obj.get('geometry_type')
                })
                
                popup_html = f"<h6>{name}</h6>"

                if description:
                    short_desc = description[:200] + "..." if len(description) > 200 else description
                    popup_html += f"<p>{short_desc}</p>"
                
                objects_for_map.append({
                    'tooltip': name,
                    'popup': popup_html,
                    'geojson': geojson
                })
            
            # Создаем карту с именем из redis_key (кешированное имя)
            map_name = redis_key.replace("cache:", "map_").replace(":", "_")
            map_result = geo.draw_custom_geometries(objects_for_map, map_name)
            
            # Подготавливаем детальную информацию с external_id (только в данных)
            detailed_objects = []
            for obj in objects:
                features = obj.get('features', {})
                external_id = extract_external_id(features)
                
                detailed_objects.append({
                    "name": obj.get('name'), 
                    "description": obj.get('description'),
                    "type": obj.get('type'),
                    "external_id": external_id,
                    "geometry_type": obj.get('geometry_type'),
                    "primary_types": obj.get('primary_types', []),
                    "specific_types": obj.get('specific_types', [])
                })
            
            map_result["count"] = len(objects)
            map_result["answer"] = answer
            map_result["objects"] = detailed_objects
            
            # ДОБАВЛЯЕМ used_objects и not_used_objects К СУЩЕСТВУЮЩЕЙ СТРУКТУРЕ
            map_result["used_objects"] = used_objects
            map_result["not_used_objects"] = []  # В прямом поиске все объекты используются
            
            # Добавляем информацию о разрешении синонимов в ответ
            if resolved_object_info and resolved_object_info.get("resolved", False):
                map_result["synonym_resolution"] = {
                    "original_name": resolved_object_info["original_name"],
                    "resolved_name": object_name,
                    "original_type": resolved_object_info.get("original_type", object_type)
                }
            
            # Добавляем статистику по external_id (только для отладки)
            objects_with_external_id = [obj for obj in detailed_objects if obj.get('external_id')]
            if debug_mode and objects_with_external_id:
                debug_info["external_id_stats"] = {
                    "total_objects": len(detailed_objects),
                    "with_external_id": len(objects_with_external_id)
                }
            
            # Добавляем debug информацию
            if debug_mode:
                debug_info["visualization"] = {
                    "map_name": map_name,
                    "total_objects_on_map": len(objects_for_map),
                    "search_type": "direct_object_search"
                }
                map_result["debug"] = debug_info

            # Сохраняем в кеш (30 минут для прямого поиска)
            set_cached_result(redis_key, map_result, expire_time=1800)
            
            return jsonify(map_result)
            
        except Exception as e:
            logger.error(f"Ошибка прямого поиска объекта: {str(e)}")
            debug_info["error"] = str(e)
            response = {"error": "Внутренняя ошибка сервера при поиске объекта"}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 500
    
    # СТАРАЯ ЛОГИКА: Поиск по области (если area_name указан)
    if not area_name:
        response = {"error": "area_name is required when no object_name provided"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400
    
    # Используем relational_service для поиска полигона области
    area_geometry = None
    area_info = None
    
    try:
        # Ищем полигон области через relational_service
        area_results = relational_service.find_area_geometry(area_name)
        
        if area_results:
            area_geometry = area_results.get("geometry")
            area_info = area_results.get("area_info", {})
            
            debug_info["steps"].append({
                "step": "area_search",
                "found": True,
                "area_title": area_info.get('title', area_name),
                "source": area_info.get('source', 'unknown')
            })
        else:
            debug_info["steps"].append({
                "step": "area_search", 
                "found": False,
                "error": "Area not found in map_content"
            })
            
    except Exception as e:
        logger.error(f"Ошибка поиска области через relational_service: {str(e)}")
        debug_info["steps"].append({
            "step": "area_search",
            "error": str(e)
        })
    
    if not area_geometry:
        response = {"error": f"Полигон для области '{area_name}' не найден в базе данных"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 404

    try:
        # Используем search_service для поиска объектов с новыми параметрами (используем разрешенные имя и тип)
        results = search_service.get_objects_in_area_by_type(
            area_geometry=area_geometry,
            object_type=object_type,
            object_subtype=object_subtype,
            object_name=object_name,
            limit=int(limit),
            search_around=search_around,
            buffer_radius_km=float(buffer_radius_km)
        )
        
        objects = results.get("objects", [])
        answer = results.get("answer", "")
        search_stats = results.get("search_stats", {})
        
        # Debug информация о результатах
        debug_info["search_results"] = {
            "total_objects_found": len(objects),
            "search_criteria": {
                "object_type": object_type,
                "object_subtype": object_subtype,
                "object_name": object_name,
                "search_around": search_around,
                "buffer_radius_km": buffer_radius_km
            },
            "location_stats": search_stats
        }
        
        if not objects:
            response = {
                "status": "no_objects", 
                "message": answer
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response)
        
        # Подготавливаем объекты для карты и собираем информацию об объектах
        objects_for_map = []
        used_objects = []
        
        # Добавляем полигон области как первый объект
        area_title = area_info.get('title', area_name) if area_info else area_name
        objects_for_map.append({
            'tooltip': f"Область поиска: {area_title}",
            'popup': f"<h6>{area_title}</h6><p>Область поиска</p>",
            'geojson': area_geometry
        })
        
        # Если включен поиск вокруг, добавляем буферную зону
        buffer_geometry = None
        if search_around:
            # Создаем буферную зону для визуализации через geo_service
            buffer_geometry = search_service.geo_service.create_buffer_geometry(area_geometry, buffer_radius_km)
            if buffer_geometry:
                objects_for_map.append({
                    'tooltip': f"Зона поиска (+{buffer_radius_km} км)",
                    'popup': f"<h6>Зона поиска</h6><p>Буферная зона {buffer_radius_km} км вокруг области</p>",
                    'geojson': buffer_geometry,
                    'style': {'color': 'orange', 'fillOpacity': 0.1, 'weight': 2}
                })
                debug_info["steps"].append({
                    "step": "buffer_zone_creation",
                    "success": True,
                    "buffer_radius_km": buffer_radius_km
                })
            else:
                debug_info["steps"].append({
                    "step": "buffer_zone_creation", 
                    "success": False,
                    "error": "Failed to create buffer geometry"
                })
        
        for obj in objects:
            name = obj.get('name', 'Без имени')
            description = obj.get('description', '')
            geojson = obj.get('geojson', {})
            location_type = obj.get('location_type', 'inside')
            
            # Добавляем объект в used_objects
            used_objects.append({
                "name": name,
                "type": obj.get('type', 'unknown'),
                "external_id": extract_external_id(obj.get('features', {})),
                "geometry_type": obj.get('geometry_type'),
                "location_type": location_type
            })
            
            popup_html = f"<h6>{name}</h6>"
            if description:
                short_desc = description[:200] + "..." if len(description) > 200 else description
                popup_html += f"<p>{short_desc}</p>"
            
            objects_for_map.append({
                'tooltip': name,
                'popup': popup_html,
                'geojson': geojson
            })
        
        # Создаем карту с именем из redis_key (кешированное имя)
        map_name = redis_key.replace("cache:", "map_").replace(":", "_")
        map_result = geo.draw_custom_geometries(objects_for_map, map_name)
        
        # Подготавливаем детальную информацию об объектах
        detailed_objects = []
        for obj in objects:
            features = obj.get('features', {})
            external_id = extract_external_id(features)
            
            detailed_objects.append({
                "name": obj.get('name'), 
                "description": obj.get('description'),
                "type": obj.get('type'),
                "external_id": external_id,
                "geometry_type": obj.get('geometry_type'),
                "primary_types": obj.get('primary_types', []),
                "specific_types": obj.get('specific_types', []),
                "location_type": obj.get('location_type', 'inside')
            })
        
        map_result["count"] = len(objects)
        map_result["answer"] = answer
        map_result["objects"] = detailed_objects
        map_result["search_stats"] = search_stats
        
        # ДОБАВЛЯЕМ used_objects и not_used_objects К СУЩЕСТВУЮЩЕЙ СТРУКТУРЕ
        map_result["used_objects"] = used_objects
        map_result["not_used_objects"] = []  # В этом эндпоинте все найденные объекты используются
        
        # Добавляем информацию о буферной зоне в ответ
        if buffer_geometry:
            map_result["buffer_zone"] = {
                "radius_km": buffer_radius_km,
                "geometry": buffer_geometry
            }
        
        # Добавляем информацию о разрешении синонимов в ответ
        if resolved_object_info and resolved_object_info.get("resolved", False):
            map_result["synonym_resolution"] = {
                "original_name": resolved_object_info["original_name"],
                "resolved_name": object_name,
                "original_type": resolved_object_info.get("original_type", object_type)
            }
        
        # Добавляем статистику по external_id (только для отладки)
        objects_with_external_id = [obj for obj in detailed_objects if obj.get('external_id')]
        if debug_mode and objects_with_external_id:
            debug_info["external_id_stats"] = {
                "total_objects": len(detailed_objects),
                "with_external_id": len(objects_with_external_id)
            }
        
        # Добавляем debug информацию
        if debug_mode:
            debug_info["visualization"] = {
                "map_name": map_name,
                "total_objects_on_map": len(objects_for_map),
                "area_included": True,
                "buffer_zone_included": search_around and buffer_geometry is not None,
                "objects_inside": search_stats.get('inside_area', 0),
                "objects_around": search_stats.get('around_area', 0)
            }
            map_result["debug"] = debug_info

        # Сохраняем в кеш (1 час для поиска по области)
        set_cached_result(redis_key, map_result, expire_time=3600)
        
        return jsonify(map_result)
        
    except Exception as e:
        logger.error(f"Ошибка поиска объектов по типу в области: {str(e)}")
        debug_info["error"] = str(e)
        response = {"error": "Внутренняя ошибка сервера при поиске"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 500
           
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
    
@app.route("/search_images_by_features", methods=["POST"])
def search_images_by_features():
    """
    Поиск изображений по признакам из feature_data
    Можно искать как по виду, так и только по признакам
    """
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    in_stoplist = request.args.get("in_stoplist", "1")  # Новый параметр
    
    debug_info = {
        "timestamp": time.time(),
        "steps": []
    }
    
    try:
        data = request.get_json()
        species_name = data.get("species_name")
        features = data.get("features", {})
        if "fruits_present" not in features:
            features["fruits_present"] = "нет"
            
        if not species_name and not features:
            response = {
                "error": "Необходимо указать species_name или features",
                "used_objects": [],
                "not_used_objects": []
            }
            return jsonify(response), 400
        
        logger.info(f"🔍 /search_images_by_features - получен запрос с параметрами:")
        logger.info(f"   - species_name: {data.get('species_name')}")
        logger.info(f"   - features: {data.get('features', {})}")
        logger.info(f"   - query_params: debug_mode={debug_mode}, in_stoplist={in_stoplist}")
        logger.info(f"   - raw_data: {data}")
        
        # Debug информация о запросе
        debug_info["parameters"] = {
            "species_name": species_name,
            "features": features,
            "in_stoplist": in_stoplist,
            "timestamp": time.time()
        }
        
        # ИНИЦИАЛИЗИРУЕМ ПЕРЕМЕННЫЕ ЗАРАНЕЕ
        safe_images = []
        stoplisted_images = []
        result = None
        
        if species_name:
            result = search_service.search_images_by_features(
                species_name=species_name,
                features=features
            )
            
            # ФИЛЬТРАЦИЯ ПО STOPLIST для изображений
            if result.get("status") == "success" and "images" in result:
                safe_images = []
                stoplisted_images = []
                
                for image in result["images"]:
                    feature_data = image.get("features", {})
                    image_in_stoplist = feature_data.get("in_stoplist")
                    
                    # Проверяем уровень безопасности
                    try:
                        requested_level = int(in_stoplist)
                        if image_in_stoplist is None or int(image_in_stoplist) <= requested_level:
                            safe_images.append(image)
                        else:
                            stoplisted_images.append(image)
                            logger.info(f"Исключено изображение с in_stoplist={image_in_stoplist}: {image.get('title', 'Без названия')}")
                    except (ValueError, TypeError):
                        # Если ошибка преобразования, используем уровень по умолчанию (1)
                        if image_in_stoplist is None or int(image_in_stoplist) <= 1:
                            safe_images.append(image)
                        else:
                            stoplisted_images.append(image)
                            logger.info(f"Исключено изображение с in_stoplist={image_in_stoplist}: {image.get('title', 'Без названия')}")
                
                # Обновляем результат с безопасными изображениями
                result["images"] = safe_images
                result["count"] = len(safe_images)
                
                # Добавляем информацию о фильтрации
                result["in_stoplist_filter_applied"] = True
                result["in_stoplist_level"] = in_stoplist
                result["stoplisted_count"] = len(stoplisted_images)
            
            # ============================================================================
            # ФОРМИРОВАНИЕ used_objects И not_used_objects ДЛЯ ПОИСКА ПО ВИДУ
            # ============================================================================
            used_objects = []      # Объекты, соответствующие найденным изображениям
            not_used_objects = []  # Объекты, не соответствующие критериям (в этом эндпоинте всегда пусто)
            
            if result.get("status") == "success" and result.get("images"):
                # Для поиска по виду - used_objects содержит основной вид
                used_objects.append({
                    "name": species_name,
                    "type": "biological_entity",
                    "images_count": len(result["images"])
                })
            
            # Добавляем объекты в результат
            result["used_objects"] = used_objects
            result["not_used_objects"] = not_used_objects
            
            # Добавляем debug информацию
            if debug_mode:
                debug_info["search_type"] = "with_species"
                debug_info["synonyms_used"] = result.get("synonyms_used", {})
                debug_info["database_query"] = {
                    "species_conditions": result.get("species_conditions", []),
                    "feature_conditions": list(features.keys())
                }
                debug_info["stoplist_filter"] = {
                    "total_before_filter": len(result.get("images", [])),
                    "safe_after_filter": len(safe_images),
                    "stoplisted_count": len(stoplisted_images)
                }
                result["debug"] = debug_info
                
            if result.get("status") == "not_found":
                # Добавляем пустые объекты для случая "не найдено"
                result["used_objects"] = []
                result["not_used_objects"] = []
                return jsonify(result), 404
            return jsonify(result)
        
        else:
            # Поиск только по признакам (без указания вида)
            result = search_service.relational_service.search_images_by_features_only(
                features=features
            )
            
            # ФИЛЬТРАЦИЯ ПО STOPLIST для изображений (только по признакам)
            if result.get("status") == "success" and "images" in result:
                safe_images = []
                stoplisted_images = []
                
                for image in result["images"]:
                    feature_data = image.get("features", {})
                    image_in_stoplist = feature_data.get("in_stoplist")
                    
                    try:
                        requested_level = int(in_stoplist)
                        if image_in_stoplist is None or int(image_in_stoplist) <= requested_level:
                            safe_images.append(image)
                        else:
                            stoplisted_images.append(image)
                            logger.info(f"Исключено изображение с in_stoplist={image_in_stoplist}: {image.get('title', 'Без названия')}")
                    except (ValueError, TypeError):
                        if image_in_stoplist is None or int(image_in_stoplist) <= 1:
                            safe_images.append(image)
                        else:
                            stoplisted_images.append(image)
                            logger.info(f"Исключено изображение с in_stoplist={image_in_stoplist}: {image.get('title', 'Без названия')}")
                
                result["images"] = safe_images
                result["count"] = len(safe_images)
                result["in_stoplist_filter_applied"] = True
                result["in_stoplist_level"] = in_stoplist
                result["stoplisted_count"] = len(stoplisted_images)
            
            # ============================================================================
            # ФОРМИРОВАНИЕ used_objects И not_used_objects ДЛЯ ПОИСКА ТОЛЬКО ПО ПРИЗНАКАМ
            # ============================================================================
            used_objects = []      # Виды, соответствующие найденным изображениям
            not_used_objects = []  # Всегда пустой массив
            
            if result.get("status") == "success" and result.get("images"):
                # Собираем уникальные виды из найденных изображений
                unique_species = {}
                for image in result["images"]:
                    species = image.get("species_name")
                    if species and species not in unique_species:
                        unique_species[species] = {
                            "name": species,
                            "type": "biological_entity",
                            "images_count": 0
                        }
                    if species:
                        unique_species[species]["images_count"] += 1
                
                used_objects = list(unique_species.values())
            
            # Добавляем объекты в результат
            result["used_objects"] = used_objects
            result["not_used_objects"] = not_used_objects
            
            # Добавляем debug информацию
            if debug_mode:
                debug_info["search_type"] = "features_only"
                debug_info["database_query"] = {
                    "feature_conditions": list(features.keys())
                }
                debug_info["stoplist_filter"] = {
                    "total_before_filter": len(result.get("images", [])),
                    "safe_after_filter": len(safe_images),
                    "stoplisted_count": len(stoplisted_images)
                }
                result["debug"] = debug_info
                
            if result.get("status") == "not_found":
                # Добавляем пустые объекты для случая "не найдено"
                result["used_objects"] = []
                result["not_used_objects"] = []
                return jsonify(result), 404
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Ошибка поиска изображений по признакам: {str(e)}")
        error_response = {
            "status": "error",
            "message": f"Ошибка при поиске изображений: {str(e)}",
            "used_objects": [],
            "not_used_objects": []
        }
        if debug_mode:
            debug_info["error"] = str(e)
            error_response["debug"] = debug_info
        return jsonify(error_response), 500
    
@app.route("/object/description/", methods=["GET", "POST"])
def get_object_description():
    # Обработка GET параметров
    logger.info(f"📦 /object/description - GET params: {dict(request.args)}")
    logger.info(f"📦 /object/description - POST data: {request.get_json()}")
    
    object_name = request.args.get("object_name")
    query = request.args.get("query")
    limit = int(request.args.get("limit", 1500))
    similarity_threshold = float(request.args.get("similarity_threshold", 0.35))
    include_similarity = request.args.get("include_similarity", "false").lower() == "true"
    use_gigachat_filter = request.args.get("use_gigachat_filter", "false").lower() == "true"
    use_gigachat_answer = request.args.get("use_gigachat_answer", "false").lower() == "true"
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    object_type = request.args.get("object_type", "all")
    save_prompt = request.args.get("save_prompt", "false").lower() == "true"
    in_stoplist = request.args.get("in_stoplist", "1")

    # Обработка POST body
    filter_data = None
    if request.method == "POST" and request.is_json:
        filter_data = request.get_json()
        logger.debug(f"Получены фильтры из body: {filter_data}")

    # Debug информация
    debug_info = {
        "parameters": {
            "object_name": object_name,
            "object_type": object_type,
            "query": query,
            "limit": limit,
            "similarity_threshold": similarity_threshold,
            "include_similarity": include_similarity,
            "use_gigachat_filter": use_gigachat_filter,
            "use_gigachat_answer": use_gigachat_answer,
            "filter_data": filter_data,
            "save_prompt": save_prompt,
            "in_stoplist": in_stoplist
        },
        "timestamp": time.time(),
        "steps": []
    }

    # РАЗРЕШЕНИЕ СИНОНИМОВ ОБЪЕКТОВ
    resolved_object_info = None
    if object_name:
        resolved_object_info = search_service.resolve_object_synonym(object_name, object_type)
        
        debug_info["synonym_resolution"] = {
            "original_name": object_name,
            "original_type": object_type,
            "resolved_info": resolved_object_info
        }
        
        if resolved_object_info.get("resolved", False):
            object_name = resolved_object_info["main_form"]
            # Не меняем object_type, если он не был передан изначально
            if object_type != "all":
                object_type = resolved_object_info["object_type"]
            logger.info(f"✅ Разрешен синоним объекта: '{resolved_object_info['original_name']}' -> '{object_name}' (тип: {object_type})")
        else:
            logger.info(f"ℹ️ Синоним для объекта '{object_name}' не найден, используем оригинальное название")

    # ВАЖНО: Если use_gigachat_answer=True, то query обязателен
    if use_gigachat_answer and not query:
        response = {"error": "Параметр 'query' обязателен при use_gigachat_answer=true"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400

    # Если нет ни object_name, ни query, ни filter_data - возвращаем ошибку
    if not object_name and not query and not filter_data:
        response = {"error": "Необходимо указать object_name, query или передать фильтры в body"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400

    # Вспомогательная функция для извлечения external_id
    def extract_external_id(desc_data):
        """Упрощенная функция извлечения external_id из данных описания"""
        if not desc_data or not isinstance(desc_data, dict):
            return None
        
        # Основной путь: structured_data -> metadata -> meta_info -> id
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

    # НОВАЯ ФУНКЦИЯ: Правильное формирование заголовка
    def get_proper_title(desc, fallback_name=None, index=1):
        """
        Формирует корректный заголовок в порядке приоритета:
        1. object_name из БД
        2. title из feature_data
        3. Первая строка content (как крайний вариант)
        4. Заголовок по умолчанию
        """
        if not isinstance(desc, dict):
            return f"Описание {index}"
        
        # 1. Приоритет: object_name из базы данных
        title = desc.get("object_name")
        if title and title.strip():
            return title.strip()
        
        # 2. Заголовок из feature_data
        feature_data = desc.get("feature_data", {})
        if isinstance(feature_data, dict):
            title = feature_data.get("title")
            if title and title.strip():
                return title.strip()
        
        # 3. Заголовок из structured_data
        structured_data = desc.get("structured_data", {})
        if isinstance(structured_data, dict):
            # Проверяем различные возможные пути в structured_data
            metadata = structured_data.get("metadata", {})
            if isinstance(metadata, dict):
                meta_info = metadata.get("meta_info", {})
                if isinstance(meta_info, dict):
                    title = meta_info.get("title")
                    if title and title.strip():
                        return title.strip()
            
            # Проверяем корневые поля structured_data
            title = structured_data.get("title")
            if title and title.strip():
                return title.strip()
        
        # 4. Первая строка content (как крайний вариант)
        content = desc.get("content", "")
        if content and isinstance(content, str):
            lines = content.strip().split('\n')
            if lines and lines[0].strip():
                first_line = lines[0].strip()
                # Обрезаем слишком длинные строки
                if len(first_line) > 100:
                    return first_line[:97] + "..."
                return first_line
        
        # 5. Заголовок по умолчанию с использованием имени объекта
        if fallback_name and fallback_name.strip():
            return f"{fallback_name} - описание {index}"
        
        return f"Описание {index}"

    try:
        # Определяем лимиты для разных случаев
        search_limit = limit if limit > 0 else 1500
        context_limit = 6
        
        if filter_data:
            descriptions = search_service.get_object_descriptions_by_filters(
                filter_data=filter_data,
                object_type=object_type,
                limit=search_limit,
                in_stoplist=in_stoplist,
                object_name=object_name  # Передаем разрешенное название для точного поиска
            )
            search_method = "filter_search"
            
        elif query:
            embedding = search_service.embedding_model.embed_query(query)
            
            if not isinstance(embedding, list):
                logger.error(f"Embedding должен быть списком, получен: {type(embedding)}")
                return jsonify({"error": "Internal embedding error"}), 500
                
            if not all(isinstance(x, (int, float)) for x in embedding):
                logger.error("Embedding содержит нечисловые элементы")
                return jsonify({"error": "Internal embedding error"}), 500
                
            if object_name:
                descriptions = search_service.get_object_descriptions_with_embedding(
                    object_name=object_name,
                    object_type=object_type,
                    query_embedding=embedding,
                    limit=search_limit,
                    similarity_threshold=similarity_threshold,
                    in_stoplist=in_stoplist
                )
                search_method = "object_with_embedding"
            else:
                descriptions = search_service.search_objects_by_embedding(
                    query_embedding=embedding,
                    object_type=object_type,
                    limit=search_limit,
                    similarity_threshold=similarity_threshold,
                    in_stoplist=in_stoplist
                )
                search_method = "semantic_search"
                
        else:
            descriptions_text = search_service.get_object_descriptions(
                object_name, 
                object_type,
                in_stoplist=in_stoplist
            )
            
            if include_similarity:
                descriptions = [{"content": text, "similarity": None, "source": "content"} 
                              for text in descriptions_text]
            else:
                descriptions = [{"content": text, "source": "content"} 
                              for text in descriptions_text]
            search_method = "simple_search"

        # Debug информация о результатах поиска
        if debug_mode:
            debug_info["search_method"] = search_method
            debug_info["search_results"] = {
                "total_found": len(descriptions),
                "search_limit": search_limit,
                "similarities": [desc.get("similarity", 0) for desc in descriptions] if descriptions and search_method != "simple_search" else []
            }

        # Проверяем, есть ли безопасные записи (с подходящим in_stoplist)
        safe_descriptions = []
        stoplisted_descriptions = []

        for desc in descriptions:
            if isinstance(desc, dict):
                feature_data = desc.get("feature_data", {})
                desc_in_stoplist = feature_data.get("in_stoplist") if feature_data else None
                
                try:
                    requested_level = int(in_stoplist)
                    if desc_in_stoplist is None or int(desc_in_stoplist) <= requested_level:
                        safe_descriptions.append(desc)
                    else:
                        stoplisted_descriptions.append(desc)
                except (ValueError, TypeError):
                    if desc_in_stoplist is None or int(desc_in_stoplist) <= 1:
                        safe_descriptions.append(desc)
                    else:
                        stoplisted_descriptions.append(desc)
            else:
                safe_descriptions.append(desc)
                
        if debug_mode and descriptions:
            debug_info["sample_description_structure"] = []
            for i, desc in enumerate(descriptions[:2]):
                if isinstance(desc, dict):
                    sample_structure = {
                        "index": i,
                        "keys": list(desc.keys()),
                        "has_feature_data": 'feature_data' in desc,
                        "has_structured_data": 'structured_data' in desc,
                        "object_name": desc.get("object_name"),
                        "calculated_title": get_proper_title(desc, object_name, i+1)
                    }
                    if 'feature_data' in desc and isinstance(desc['feature_data'], dict):
                        sample_structure["feature_data_keys"] = list(desc['feature_data'].keys())
                        if 'metadata' in desc['feature_data'] and isinstance(desc['feature_data']['metadata'], dict):
                            sample_structure["metadata_keys"] = list(desc['feature_data']['metadata'].keys())
                    debug_info["sample_description_structure"].append(sample_structure)
        
        # Debug информация о фильтрации in_stoplist
        if debug_mode:
            debug_info["in_stoplist_filter"] = {
                "total_before_filter": len(descriptions),
                "safe_after_filter": len(safe_descriptions),
                "stoplisted_count": len(stoplisted_descriptions),
                "requested_level": in_stoplist
            }

        # Если после фильтрации не осталось безопасных документов
        if not safe_descriptions:
            response = {"error": "Я не готов про это разговаривать"}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 400

        # Используем только безопасные описания для дальнейшей обработки
        descriptions = safe_descriptions

        # Обработка use_gigachat_filter
        if use_gigachat_filter:
            filter_query = query if query else object_name
            
            if debug_mode:
                debug_info["before_gigachat_filter"] = {
                    "count": len(descriptions),
                    "filter_query": filter_query
                }
            
            filtered_descriptions = search_service.filter_text_descriptions_with_gigachat(
                filter_query, 
                descriptions
            )
            
            if debug_mode:
                debug_info["after_gigachat_filter"] = {
                    "count": len(filtered_descriptions),
                    "filtered_out": len(descriptions) - len(filtered_descriptions)
                }

            descriptions = filtered_descriptions

        # ============================================================================
        # ФОРМИРОВАНИЕ used_objects И not_used_objects ДЛЯ РАЗНЫХ СЦЕНАРИЕВ
        # ============================================================================
        
        used_objects = []      # Объекты, использованные в контексте GigaChat
        not_used_objects = []  # Объекты, не вошедшие в контекст GigaChat

        # Обработка use_gigachat_answer
        if use_gigachat_answer:
            if not descriptions:
                response = {"error": "Не найдено описаний для генерации ответа"}
                if debug_mode:
                    response["debug"] = debug_info
                return jsonify(response), 404

            # ФИЛЬТРАЦИЯ BLACKLIST_RISK
            safe_descriptions_for_gigachat = []
            blacklisted_descriptions = []
            
            for desc in descriptions:
                if isinstance(desc, dict):
                    feature_data = desc.get("feature_data", {})
                    if feature_data and feature_data.get("blacklist_risk") is True:
                        blacklisted_descriptions.append(desc)
                        continue
                    
                    if desc.get("blacklist_risk") is True:
                        blacklisted_descriptions.append(desc)
                        continue
                
                safe_descriptions_for_gigachat.append(desc)
            
            # Debug информация о фильтрации blacklist
            if debug_mode:
                debug_info["blacklist_filter"] = {
                    "total_before_filter": len(descriptions),
                    "safe_after_filter": len(safe_descriptions_for_gigachat),
                    "blacklisted_count": len(blacklisted_descriptions)
                }
            
            # Если после фильтрации не осталось безопасных документов
            if not safe_descriptions_for_gigachat:
                response = {"error": "Все описания содержат риск blacklist и не могут быть использованы для генерации ответа GigaChat"}
                if debug_mode:
                    response["debug"] = debug_info
                return jsonify(response), 400

            descriptions_for_context = safe_descriptions_for_gigachat

            # Берем топ безопасных описаний для контекста
            if all('similarity' in desc for desc in descriptions_for_context):
                context_descriptions = sorted(descriptions_for_context, key=lambda x: x.get('similarity', 0), reverse=True)[:context_limit]
            else:
                context_descriptions = descriptions_for_context[:context_limit]
            
            # ============================================================================
            # ФОРМИРОВАНИЕ СПИСКОВ ОБЪЕКТОВ ДЛЯ СЦЕНАРИЯ С GIGACHAT
            # ============================================================================
            
            # used_objects - объекты из контекста GigaChat (топ по релевантности)
            for desc in context_descriptions:
                if isinstance(desc, dict):
                    obj_info = {
                        "name": desc.get("object_name", object_name if object_name else "semantic_search"),
                        "type": desc.get("object_type", object_type),
                        "source": desc.get("source", "unknown"),
                        "similarity": round(desc.get("similarity", 0), 4) if desc.get("similarity") else None
                    }
                    used_objects.append(obj_info)
            
            # not_used_objects - объекты, не вошедшие в контекст GigaChat
            remaining_descriptions = [desc for desc in descriptions_for_context if desc not in context_descriptions]
            for desc in remaining_descriptions:
                if isinstance(desc, dict):
                    obj_info = {
                        "name": desc.get("object_name", object_name if object_name else "semantic_search"),
                        "type": desc.get("object_type", object_type),
                        "source": desc.get("source", "unknown"),
                        "similarity": round(desc.get("similarity", 0), 4) if desc.get("similarity") else None
                    }
                    not_used_objects.append(obj_info)
            
            # Объединяем топ безопасных описаний в контекст
            context = "\n\n".join([
                desc["content"] if isinstance(desc, dict) else desc 
                for desc in context_descriptions
            ])

            # Добавляем информацию о количестве найденных записей
            total_count = len(descriptions_for_context)
            count_info = f"\n\nВсего найдено безопасных записей: {total_count}"
            if len(blacklisted_descriptions) > 0:
                count_info += f" (исключено {len(blacklisted_descriptions)} записей с риском blacklist)"
            if total_count > context_limit:
                count_info += f" (в контекст включено топ-{context_limit} по релевантности)"
            logger.debug(context)
            context += count_info
            
            # СОХРАНЕНИЕ ПОЛНОГО ПРОМПТА
            full_prompt = f"""Ты эксперт по Байкальской природной территории. 
            Используй твою базу знаний для точных ответов на вопросы пользователя.

            Особые указания:
            - На вопросы 'сколько' - подсчитай количество соответствующих записей в базе знаний
            Например, на вопрос 'Сколько музеев?' при информации 'Всего найдено записей: 98 (в контекст включено топ-5 по релевантности)', нужно ответить около 98 музеев и затем описание каждого музея из топ записей
            - Будь информативным и лаконичным
            - Даже при неполной информации предоставь доступные детали

            Твоя база знаний:
            {context}

            Вопрос: {query}

            Ответ:"""
            
            if save_prompt:
                current_dir = Path(__file__).parent
                timestamp = int(time.time())
                prompt_filename = current_dir / f"gigachat_prompt_{timestamp}.txt"
                
                try:
                    with open(prompt_filename, 'w', encoding='utf-8') as f:
                        f.write(full_prompt)
                    logger.info(f"✅ Полный промпт сохранен в: {prompt_filename}")
                except Exception as e:
                    logger.error(f"❌ Ошибка сохранения промпта: {e}")
            
            # Генерируем ответ с помощью GigaChat
            try:
                gigachat_result = search_service._generate_gigachat_answer(query, context)
                
                # Проверяем, был ли ответ заблокирован
                is_blacklist = gigachat_result.get("finish_reason") == "blacklist" or not gigachat_result.get("success", True)
                
                # Если ответ заблокирован, возвращаем форматированные безопасные описания
                if is_blacklist:
                    logger.info("🚫 GigaChat вернул blacklist, возвращаем форматированные безопасные описания")
                    
                    # Форматируем безопасные описания с ПРАВИЛЬНЫМИ ЗАГОЛОВКАМИ
                    formatted_descriptions = []
                    for i, desc in enumerate(descriptions_for_context, 1):
                        if isinstance(desc, dict):
                            content = desc.get("content", "")
                            similarity = desc.get("similarity")
                            source = desc.get("source", "unknown")
                            
                            # ИЗВЛЕКАЕМ EXTERNAL_ID (только для данных)
                            external_id = extract_external_id(desc)
                            
                            # ИСПОЛЬЗУЕМ ПРАВИЛЬНУЮ ФУНКЦИЮ ДЛЯ ЗАГОЛОВКА
                            title = get_proper_title(desc, object_name, i)
                            
                            formatted_desc = {
                                "id": i,
                                "title": title,  # ПРАВИЛЬНЫЙ ЗАГОЛОВОК
                                "content": content,
                                "source": source,
                                "feature_data": desc.get("feature_data", {}),
                                "structured_data": desc.get("structured_data", {})
                            }
                            
                            # ДОБАВЛЯЕМ EXTERNAL_ID В ДАННЫЕ
                            if external_id:
                                formatted_desc["external_id"] = external_id
                            
                            if similarity is not None:
                                formatted_desc["similarity"] = round(similarity, 4)
                                
                            formatted_descriptions.append(formatted_desc)
                        else:
                            formatted_descriptions.append({
                                "id": i,
                                "title": get_proper_title(None, object_name, i),  # ЗАГОЛОВОК ПО УМОЛЧАНИЮ
                                "content": desc,
                                "source": "content"
                            })

                    # Сортируем по similarity если есть
                    if all('similarity' in desc for desc in formatted_descriptions):
                        formatted_descriptions.sort(key=lambda x: x.get('similarity', 0), reverse=True)

                    response_data = {
                        "count": len(formatted_descriptions),
                        "descriptions": formatted_descriptions,
                        "query_used": query if query else "simple_search",
                        "similarity_threshold": similarity_threshold if query else None,
                        "use_gigachat_filter": use_gigachat_filter,
                        "use_gigachat_answer": True,
                        "gigachat_restricted": True,
                        "message": "GigaChat не смог сгенерировать ответ, поэтому возвращены исходные безопасные описания",
                        "formatted": True,
                        "in_stoplist_filter_applied": True,
                        "in_stoplist_level": in_stoplist,
                        # ДОБАВЛЯЕМ ОБЪЕКТЫ
                        "used_objects": used_objects,
                        "not_used_objects": not_used_objects
                    }

                    if object_name:
                        response_data["object_name"] = object_name
                        response_data["object_type"] = object_type

                    if filter_data:
                        response_data["filters_applied"] = filter_data

                    # Добавляем информацию о разрешении синонимов
                    if resolved_object_info and resolved_object_info.get("resolved", False):
                        response_data["synonym_resolution"] = {
                            "original_name": resolved_object_info["original_name"],
                            "resolved_name": object_name,
                            "original_type": resolved_object_info.get("original_type", object_type)
                        }

                    if debug_mode:
                        response_data["debug"] = debug_info
                        response_data["debug"]["gigachat_generation"] = {
                            "finish_reason": gigachat_result.get("finish_reason"),
                            "blacklist_detected": True,
                            "fallback_to_descriptions": True,
                            "prompt_saved": save_prompt
                        }

                    return jsonify(response_data)
                
                # Если ответ не заблокирован, возвращаем обычный ответ GigaChat
                gigachat_response = gigachat_result.get("content", "")

                # СОБИРАЕМ EXTERNAL_ID ИЗ КОНТЕКСТНЫХ ОПИСАНИЙ
                external_ids = []
                source_descriptions_summary = []

                for desc in context_descriptions:
                    if isinstance(desc, dict):
                        # ИЗВЛЕКАЕМ EXTERNAL_ID
                        external_id = extract_external_id(desc)
                        
                        # ИСПОЛЬЗУЕМ ПРАВИЛЬНУЮ ФУНКЦИЮ ДЛЯ ЗАГОЛОВКА
                        title = get_proper_title(desc, object_name, len(source_descriptions_summary) + 1)
                        
                        desc_summary = {
                            "title": title,  # ПРАВИЛЬНЫЙ ЗАГОЛОВОК
                            "content_preview": desc.get("content", "")[:200] + "..." if len(desc.get("content", "")) > 200 else desc.get("content", ""),
                            "source": desc.get("source", "unknown"),
                            "similarity": round(desc.get("similarity", 0), 4) if desc.get("similarity") else None
                        }
                        
                        if external_id:
                            desc_summary["external_id"] = external_id
                            if external_id not in external_ids:
                                external_ids.append(external_id)
                                
                        source_descriptions_summary.append(desc_summary)

                response_data = {
                    "gigachat_answer": gigachat_response,
                    "external_ids": external_ids,  # СПИСОК ВСЕХ EXTERNAL_ID
                    "source_descriptions": source_descriptions_summary,  # КРАТКАЯ ИНФОРМАЦИЯ ОБ ИСТОЧНИКАХ
                    "context_used": {
                        "descriptions_count": len(context_descriptions),
                        "total_descriptions": total_count,
                        "blacklisted_excluded": len(blacklisted_descriptions),
                        "external_ids_count": len(external_ids)
                    },
                    "query": query,
                    "object_name": object_name if object_name else "semantic_search",
                    "object_type": object_type,
                    "in_stoplist_level": in_stoplist,
                    # ДОБАВЛЯЕМ ОБЪЕКТЫ
                    "used_objects": used_objects,
                    "not_used_objects": not_used_objects
                }
                
                # Добавляем информацию о разрешении синонимов
                if resolved_object_info and resolved_object_info.get("resolved", False):
                    response_data["synonym_resolution"] = {
                        "original_name": resolved_object_info["original_name"],
                        "resolved_name": object_name,
                        "original_type": resolved_object_info.get("original_type", object_type)
                    }
                
                if debug_mode:
                    response_data["debug"] = debug_info
                    response_data["debug"]["gigachat_generation"] = {
                        "response_length": len(gigachat_response),
                        "finish_reason": gigachat_result.get("finish_reason"),
                        "blacklist_detected": False,
                        "prompt_saved": save_prompt
                    }

                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"Ошибка генерации ответа GigaChat: {str(e)}")
                error_response = {"error": "Ошибка генерации ответа GigaChat"}
                if debug_mode:
                    debug_info["gigachat_error"] = str(e)
                    error_response["debug"] = debug_info
                return jsonify(error_response), 500

        # ============================================================================
        # ФОРМИРОВАНИЕ СПИСКОВ ОБЪЕКТОВ ДЛЯ СЦЕНАРИЯ БЕЗ GIGACHAT
        # ============================================================================
        
        # Для сценария без GigaChat:
        # used_objects - все найденные объекты (так как они все "используются" в ответе)
        # not_used_objects - пустой массив
        
        for desc in descriptions:
            if isinstance(desc, dict):
                obj_info = {
                    "name": desc.get("object_name", object_name if object_name else "semantic_search"),
                    "type": desc.get("object_type", object_type),
                    "source": desc.get("source", "unknown"),
                    "similarity": round(desc.get("similarity", 0), 4) if desc.get("similarity") else None
                }
                used_objects.append(obj_info)

        # Форматированный ответ без GigaChat
        if not descriptions:
            response = {"error": "Я не готов про это разговаривать"}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 404

        # Форматируем описания с ПРАВИЛЬНЫМИ ЗАГОЛОВКАМИ
        formatted_descriptions = []
        for i, desc in enumerate(descriptions, 1):
            if isinstance(desc, dict):
                content = desc.get("content", "")
                similarity = desc.get("similarity")
                source = desc.get("source", "unknown")
                
                # ИЗВЛЕКАЕМ EXTERNAL_ID (только для данных)
                external_id = extract_external_id(desc)
                
                # ИСПОЛЬЗУЕМ ПРАВИЛЬНУЮ ФУНКЦИЮ ДЛЯ ЗАГОЛОВКА
                title = get_proper_title(desc, object_name, i)
                
                formatted_desc = {
                    "id": i,
                    "title": title,  # ПРАВИЛЬНЫЙ ЗАГОЛОВОК
                    "content": content,
                    "source": source,
                    "feature_data": desc.get("feature_data", {}),
                    "structured_data": desc.get("structured_data", {})
                }
                
                # ДОБАВЛЯЕМ EXTERNAL_ID В ДАННЫЕ
                if external_id:
                    formatted_desc["external_id"] = external_id
                
                if similarity is not None:
                    formatted_desc["similarity"] = round(similarity, 4)
                    
                formatted_descriptions.append(formatted_desc)
            else:
                formatted_descriptions.append({
                    "id": i,
                    "title": get_proper_title(None, object_name, i),  # ЗАГОЛОВОК ПО УМОЛЧАНИЮ
                    "content": desc,
                    "source": "content"
                })

        # Сортируем по similarity если есть
        if all('similarity' in desc for desc in formatted_descriptions):
            formatted_descriptions.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        response_data = {
            "count": len(formatted_descriptions),
            "descriptions": formatted_descriptions,
            "query_used": query if query else "simple_search",
            "similarity_threshold": similarity_threshold if query else None,
            "use_gigachat_filter": use_gigachat_filter,
            "in_stoplist_filter_applied": True,
            "in_stoplist_level": in_stoplist,
            "formatted": True,
            # ДОБАВЛЯЕМ ОБЪЕКТЫ
            "used_objects": used_objects,
            "not_used_objects": []  # В сценарии без GigaChat все объекты используются
        }

        # Добавляем информацию об объекте
        if object_name:
            response_data["object_name"] = object_name
            response_data["object_type"] = object_type

        # Добавляем информацию о фильтрах
        if filter_data:
            response_data["filters_applied"] = filter_data

        # Добавляем информацию о разрешении синонимов
        if resolved_object_info and resolved_object_info.get("resolved", False):
            response_data["synonym_resolution"] = {
                "original_name": resolved_object_info["original_name"],
                "resolved_name": object_name,
                "original_type": resolved_object_info.get("original_type", object_type)
            }

        # Добавляем debug информацию
        if debug_mode:
            response_data["debug"] = debug_info

        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Ошибка получения описания: {str(e)}", exc_info=True)
        error_response = {"error": "Внутренняя ошибка сервера"}
        if debug_mode:
            debug_info["error"] = str(e)
            error_response["debug"] = debug_info
        return jsonify(error_response), 500
       
@app.route("/species/description/", methods=["GET"])
def get_species_description():
    logger.info(f"📦 /species/description - GET params: {dict(request.args)}")
    species_name = request.args.get("species_name")
    query = request.args.get("query")
    limit = int(request.args.get("limit", 1500))
    similarity_threshold = float(request.args.get("similarity_threshold", 0.1))
    include_similarity = request.args.get("include_similarity", "false").lower() == "true"
    use_gigachat_filter = request.args.get("use_gigachat_filter", "false").lower() == "true"
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    in_stoplist = request.args.get("in_stoplist", "1")

    if not species_name:
        response = {
            "error": "species_name parameter is required",
            "used_objects": [],
            "not_used_objects": []
        }
        return jsonify(response), 400

    # Debug информация
    debug_info = {
        "parameters": {
            "species_name": species_name,
            "query": query,
            "limit": limit,
            "similarity_threshold": similarity_threshold,
            "include_similarity": include_similarity,
            "use_gigachat_filter": use_gigachat_filter,
            "in_stoplist": in_stoplist
        },
        "timestamp": time.time()
    }

    try:
        if query:
            embedding = search_service.embedding_model.embed_query(query)
            
            if not isinstance(embedding, list):
                logger.error(f"Embedding должен быть списком, получен: {type(embedding)}")
                return jsonify({"error": "Internal embedding error"}), 500
                
            if not all(isinstance(x, (int, float)) for x in embedding):
                logger.error("Embedding содержит нечисловые элементы")
                return jsonify({"error": "Internal embedding error"}), 500
                
            # Debug информация об embedding
            if debug_mode:
                debug_info["embedding"] = {
                    "type": type(embedding).__name__,
                    "length": len(embedding),
                    "first_5_elements": embedding[:5] if isinstance(embedding, list) else "N/A"
                }
            
            # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ПЕРЕД ВЫЗОВОМ
            logger.info(f"🔍 ВЫЗОВ get_text_descriptions_with_embedding:")
            logger.info(f"   - species_name: {species_name}")
            logger.info(f"   - query: {query}")
            logger.info(f"   - similarity_threshold: {similarity_threshold}")
            logger.info(f"   - in_stoplist: {in_stoplist}")
            logger.info(f"   - limit: {limit}")
            
            # ИСПРАВЛЕНИЕ: Используем relational_service вместо search_service
            descriptions = search_service.relational_service.get_text_descriptions_with_embedding(
                species_name=species_name,
                query_embedding=embedding,
                limit=limit,
                similarity_threshold=similarity_threshold,
                in_stoplist=in_stoplist
            )
            
            # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ РЕЗУЛЬТАТОВ
            logger.info(f"📊 РЕЗУЛЬТАТЫ get_text_descriptions_with_embedding:")
            logger.info(f"   - Найдено описаний: {len(descriptions)}")
            
            for i, desc in enumerate(descriptions):
                if isinstance(desc, dict):
                    logger.info(f"   - Описание {i}:")
                    logger.info(f"     * similarity: {desc.get('similarity')}")
                    logger.info(f"     * object_name: {desc.get('object_name')}")
                    logger.info(f"     * has_content: {bool(desc.get('content'))}")
                    logger.info(f"     * content_length: {len(desc.get('content', ''))}")
                    logger.info(f"     * has_structured_data: {bool(desc.get('structured_data'))}")
                    logger.info(f"     * source: {desc.get('source')}")
                    if desc.get('structured_data'):
                        logger.info(f"     * structured_data_keys: {list(desc.get('structured_data', {}).keys())}")
                else:
                    logger.info(f"   - Описание {i}: тип {type(desc)}")
            
            # Debug информация о результатах поиска
            if debug_mode:
                debug_info["search_method"] = "embedding_similarity"
                debug_info["search_results"] = {
                    "total_found": len(descriptions),
                    "similarities": [desc.get("similarity", 0) for desc in descriptions] if descriptions else []
                }
            
        else:
            # ИСПРАВЛЕНИЕ: Используем search_service.get_text_descriptions
            descriptions = search_service.get_text_descriptions(species_name, in_stoplist=in_stoplist)
            
            # Debug информация
            if debug_mode:
                debug_info["search_method"] = "simple_search"
                debug_info["search_results"] = {
                    "total_found": len(descriptions)
                }
                
        # Проверяем, есть ли безопасные записи (с подходящим in_stoplist)
        safe_descriptions = []
        stoplisted_descriptions = []
        
        logger.info(f"🔒 ФИЛЬТРАЦИЯ ПО STOPLIST (уровень {in_stoplist}):")
        logger.info(f"   - Всего описаний до фильтрации: {len(descriptions)}")
        
        for desc in descriptions:
            # Проверяем feature_data на наличие in_stoplist
            if isinstance(desc, dict):
                feature_data = desc.get("feature_data", {})
                desc_in_stoplist = feature_data.get("in_stoplist") if feature_data else None
                
                # Если in_stoplist не указан или <= запрошенному уровню, считаем безопасным
                try:
                    requested_level = int(in_stoplist)
                    if desc_in_stoplist is None or int(desc_in_stoplist) <= requested_level:
                        safe_descriptions.append(desc)
                        logger.info(f"   ✓ БЕЗОПАСНО: in_stoplist={desc_in_stoplist}")
                    else:
                        stoplisted_descriptions.append(desc)
                        logger.info(f"   ✗ STOPLIST: in_stoplist={desc_in_stoplist} > запрошенного {requested_level}")
                except (ValueError, TypeError):
                    # Если ошибка преобразования, используем уровень по умолчанию (1)
                    if desc_in_stoplist is None or int(desc_in_stoplist) <= 1:
                        safe_descriptions.append(desc)
                        logger.info(f"   ✓ БЕЗОПАСНО (по умолчанию): in_stoplist={desc_in_stoplist}")
                    else:
                        stoplisted_descriptions.append(desc)
                        logger.info(f"   ✗ STOPLIST (по умолчанию): in_stoplist={desc_in_stoplist}")
            else:
                # Для простых строк считаем безопасными
                safe_descriptions.append(desc)
                logger.info(f"   ✓ БЕЗОПАСНО: простое описание")

        # Debug информация о фильтрации in_stoplist
        if debug_mode:
            debug_info["in_stoplist_filter"] = {
                "total_before_filter": len(descriptions),
                "safe_after_filter": len(safe_descriptions),
                "stoplisted_count": len(stoplisted_descriptions),
                "requested_level": in_stoplist
            }

        logger.info(f"📋 ИТОГИ ФИЛЬТРАЦИИ:")
        logger.info(f"   - Безопасные описания: {len(safe_descriptions)}")
        logger.info(f"   - Исключено по stoplist: {len(stoplisted_descriptions)}")

        # Если после фильтрации не осталось безопасных документов
        if not safe_descriptions:
            logger.warning(f"🚫 НЕТ БЕЗОПАСНЫХ ОПИСАНИЙ для '{species_name}'")
            response = {
                "error": "Я не готов про это разговаривать",
                "used_objects": [],
                "not_used_objects": []
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 400

        # Используем только безопасные описания для дальнейшей обработки
        descriptions = safe_descriptions

        # ============================================================================
        # ФОРМИРОВАНИЕ used_objects И not_used_objects
        # ============================================================================
        used_objects = []      # Все найденные объекты (в этом эндпоинте все используются)
        not_used_objects = []  # Пустой массив (все объекты используются)

        # В этом эндпоинте used_objects содержит информацию о виде
        for desc in descriptions:
            similarity = None
            if isinstance(desc, dict) and desc.get("similarity") is not None:
                try:
                    similarity_val = desc.get("similarity")
                    # ФИКС: Проверяем, что similarity валидно и не NaN
                    if similarity_val is not None and not math.isnan(float(similarity_val)):
                        similarity = round(float(similarity_val), 4)
                except (ValueError, TypeError):
                    similarity = None
            
            # Получаем имя объекта из описания или используем переданное species_name
            object_name = desc.get("object_name", species_name) if isinstance(desc, dict) else species_name
            
            used_objects.append({
                "name": object_name,
                "type": "biological_entity",
                "source": desc.get("source", "unknown") if isinstance(desc, dict) else "content",
                "similarity": similarity
            })

        if use_gigachat_filter:
            filter_query = query if query else species_name
            
            logger.debug("Описания для фильтрации с Gigachat")
            logger.debug(descriptions)
            
            # Debug информация до фильтрации
            if debug_mode:
                debug_info["before_gigachat_filter"] = {
                    "count": len(descriptions),
                    "filter_query": filter_query
                }
            
            filtered_descriptions = search_service.filter_text_descriptions_with_gigachat(
                filter_query, 
                descriptions
            )
            
            # Debug информация после фильтрации
            if debug_mode:
                debug_info["after_gigachat_filter"] = {
                    "count": len(filtered_descriptions),
                    "filtered_out": len(descriptions) - len(filtered_descriptions)
                }

            # Обновляем used_objects после фильтрации Gigachat
            if filtered_descriptions:
                used_objects = []
                for desc in filtered_descriptions:
                    object_name = desc.get("object_name", species_name) if isinstance(desc, dict) else species_name
                    used_objects.append({
                        "name": object_name,
                        "type": "biological_entity", 
                        "source": desc.get("source", "unknown") if isinstance(desc, dict) else "content",
                        "similarity": round(desc.get("similarity", 0), 4) if isinstance(desc, dict) and desc.get("similarity") else None
                    })

            descriptions = filtered_descriptions

        if not descriptions:
            logger.warning(f"🚫 ОПИСАНИЯ ОТФИЛЬТРОВАНЫ GigaChat для '{species_name}'")
            response = {
                "error": "Я не готов про это разговаривать",
                "used_objects": [],
                "not_used_objects": []
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 404

        # ============================================================================
        # ФОРМАТИРОВАНИЕ ОТВЕТА С ДОБАВЛЕНИЕМ ИМЕНИ ОБЪЕКТА В CONTENT
        # ============================================================================
        
        def format_content_with_title(desc, index):
            """Форматирует контент с добавлением заголовка в markdown формате"""
            if isinstance(desc, dict):
                content = desc.get("content", "")
                object_name = desc.get("object_name", species_name)
                
                # Создаем заголовок в markdown формате
                title_header = f"** {object_name} **\n\n"
                
                # Объединяем заголовок с контентом
                formatted_content = title_header + content
                return formatted_content
            else:
                # Для простых строк
                return f"# {species_name}\n\n{desc}"
        
        # Форматируем ответ в зависимости от параметров
        if include_similarity:
            formatted_descriptions = []
            for i, desc in enumerate(descriptions, 1):
                if isinstance(desc, dict):
                    formatted_desc = {
                        "content": format_content_with_title(desc, i),
                        "source": desc.get("source", "unknown"),
                        "feature_data": desc.get("feature_data", {}),
                        "object_name": desc.get("object_name", species_name),
                        "object_type": "biological_entity",
                        "similarity": round(desc.get("similarity", 0), 4) if desc.get("similarity") else None
                    }
                    
                    # Добавляем structured_data если есть
                    if desc.get("structured_data"):
                        formatted_desc["structured_data"] = desc.get("structured_data")
                    
                    # Добавляем species_features если есть
                    if desc.get("species_features"):
                        formatted_desc["species_features"] = desc.get("species_features")
                        
                    formatted_descriptions.append(formatted_desc)
                else:
                    formatted_descriptions.append({
                        "content": format_content_with_title(desc, i),
                        "source": "content",
                        "object_name": species_name,
                        "object_type": "biological_entity"
                    })
            
            response_data = {
                "count": len(formatted_descriptions),
                "descriptions": formatted_descriptions,
                "query_used": query if query else "simple_search",
                "similarity_threshold": similarity_threshold if query else None,
                "use_gigachat_filter": use_gigachat_filter,
                "in_stoplist_filter_applied": True,
                "in_stoplist_level": in_stoplist,
                "used_objects": used_objects,
                "not_used_objects": not_used_objects
            }
        else:
            formatted_descriptions = []
            for i, desc in enumerate(descriptions, 1):
                if isinstance(desc, dict):
                    formatted_desc = {
                        "content": format_content_with_title(desc, i),
                        "source": desc.get("source", "unknown"),
                        "feature_data": desc.get("feature_data", {}),
                        "object_name": desc.get("object_name", species_name),
                        "object_type": "biological_entity"
                    }
                    
                    # Добавляем structured_data если есть
                    if desc.get("structured_data"):
                        formatted_desc["structured_data"] = desc.get("structured_data")
                    
                    # Добавляем species_features если есть
                    if desc.get("species_features"):
                        formatted_desc["species_features"] = desc.get("species_features")
                        
                    formatted_descriptions.append(formatted_desc)
                else:
                    formatted_descriptions.append({
                        "content": format_content_with_title(desc, i),
                        "source": "content",
                        "object_name": species_name,
                        "object_type": "biological_entity"
                    })
            
            response_data = {
                "count": len(formatted_descriptions),
                "descriptions": formatted_descriptions,
                "query_used": query if query else "simple_search",
                "similarity_threshold": similarity_threshold if query else None,
                "use_gigachat_filter": use_gigachat_filter,
                "in_stoplist_filter_applied": True,
                "in_stoplist_level": in_stoplist,
                "used_objects": used_objects,
                "not_used_objects": not_used_objects
            }

        # Добавляем debug информацию
        if debug_mode:
            response_data["debug"] = debug_info

        logger.info(f"✅ УСПЕШНЫЙ ОТВЕТ для '{species_name}': {len(formatted_descriptions)} описаний")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Ошибка получения описания для '{species_name}': {str(e)}", exc_info=True)
        error_response = {
            "error": "Внутренняя ошибка сервера",
            "used_objects": [],
            "not_used_objects": []
        }
        if debug_mode:
            debug_info["error"] = str(e)
            error_response["debug"] = debug_info
        return jsonify(error_response), 500
     
@app.route("/get_coords", methods=["POST"])
def api_get_coords():
    data = request.get_json()
    name = data.get("name")
    
    logger.info(f"🔍 /get_coords - получен запрос:")
    logger.info(f"   - name: {name}")
    logger.info(f"   - raw_data: {data}")
    
    if not name:
        return jsonify({
            "status": "error", 
            "message": "Параметр 'name' обязателен.",
            "used_objects": [],
            "not_used_objects": []
        }), 400

    result = geo.get_point_coords_from_geodb(name)
    
    used_objects = []
    not_used_objects = []
    
    if result.get("status") == "ok":
        used_objects.append({
            "name": name,
            "type": "geographical_entity"
        })
    else:
        not_used_objects.append({
            "name": name, 
            "type": "geographical_entity"
        })
    
    # Обновляем результат с объектами
    result["used_objects"] = used_objects
    result["not_used_objects"] = not_used_objects
    
    return jsonify(result)

@app.route("/coords_to_map", methods=["POST"])
def api_coords_to_map():
    t0 = time.perf_counter()
    data = request.get_json()
    t_after_parse = time.perf_counter()
    lat = data.get("latitude")
    lon = data.get("longitude")
    radius = data.get("radius_km", 30)
    object_type = data.get("object_type")
    species_name = data.get("species_name")
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    in_stoplist_param = request.args.get("in_stoplist", "1")
    try:
        if in_stoplist_param.lower() in ['false', 'true']:
            in_stoplist = 1
        else:
            in_stoplist = int(in_stoplist_param)
    except (ValueError, TypeError):
        in_stoplist = 1
    
    # Параметры для кеша
    cache_params = {
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius,
        "object_type": object_type,
        "species_name": species_name,
        "in_stoplist": in_stoplist,
        "version": "v2"
    }
    
    redis_key = f"cache:coords_search:{generate_cache_key(cache_params)}"
    debug_info = {
        "timestamp": time.time(),
        "cache_key": redis_key,
        "search_time": 0,
        "parse_time": round(t_after_parse - t0, 3)
    }

    # Проверяем кеш
    cache_hit, cached_result = get_cached_result(redis_key, debug_info)
    if cache_hit:
        if debug_mode:
            cached_result["debug"] = debug_info
        return jsonify(cached_result)

    logger.debug(f"""Параметры:{data}""")
    if not lat or not lon:
        response = {
            "status": "error", 
            "message": "Не заданы координаты.",
            "used_objects": [],
            "not_used_objects": []
        }
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400

    # РАЗРЕШЕНИЕ СИНОНИМОВ ДЛЯ ВИДОВ
    resolved_species_info = None
    if species_name:
        resolved_species_info = search_service.resolve_object_synonym(species_name, "biological_entity")
        
        debug_info["species_resolution"] = {
            "original_name": species_name,
            "resolved_info": resolved_species_info
        }
        
        if resolved_species_info.get("resolved", False):
            species_name = resolved_species_info["main_form"]
            logger.info(f"✅ Разрешен синоним вида: '{resolved_species_info['original_name']}' -> '{species_name}'")
        else:
            logger.info(f"ℹ️ Синоним для вида '{species_name}' не найден, используем оригинальное название")

    # Initialize t3 in case visualization fails
    t3 = time.perf_counter()
    
    try:
        t1 = time.perf_counter()
        result = search_service.get_nearby_objects(
            latitude=float(lat),
            longitude=float(lon),
            radius_km=float(radius),
            object_type=object_type,
            species_name=species_name,
            in_stoplist=in_stoplist
        )
        t2 = time.perf_counter()
        objects = result.get("objects", [])
        answer = result.get("answer", "")
        
        # Debug информация
        debug_info["search_time"] = round(t2 - t1, 3)
        debug_info["parameters"] = {
            "latitude": lat,
            "longitude": lon,
            "radius_km": radius,
            "object_type": object_type,
            "species_name": species_name,
            "in_stoplist": in_stoplist
        }
        debug_info["objects_count"] = len(objects)
        debug_info["search_query_details"] = result.get("debug_info", {})
        
        # Добавляем информацию о разрешении синонимов в debug
        if resolved_species_info:
            debug_info["species_synonym_resolution"] = {
                "original_name": resolved_species_info.get("original_name"),
                "resolved_name": species_name,
                "resolved": resolved_species_info.get("resolved", False)
            }
        #logger.debug("Все объекты:")
        #logger.debug(objects)
        if not objects:
            response = {
                "status": "no_objects", 
                "message": answer,
                "used_objects": [],
                "not_used_objects": []
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response)

        # ФИЛЬТРАЦИЯ ПО STOPLIST для найденных объектов
        safe_objects = []
        stoplisted_objects = []
        # logger.debug("Безопасные")
        # logger.debug(safe_objects)
        # logger.debug("Стоплистед")
        # logger.debug(stoplisted_objects)
        for obj in objects:
            # Проверяем feature_data объектов на in_stoplist
            feature_data = obj.get("features", {})
            obj_in_stoplist = feature_data.get("in_stoplist")
            
            try:
                requested_level = int(in_stoplist)
                if obj_in_stoplist is None or int(obj_in_stoplist) <= requested_level:
                    safe_objects.append(obj)
                else:
                    stoplisted_objects.append(obj)
                    logger.info(f"Исключен объект с in_stoplist={obj_in_stoplist}: {obj.get('name', 'Без имени')}")
            except (ValueError, TypeError):
                if obj_in_stoplist is None or int(obj_in_stoplist) <= 1:
                    safe_objects.append(obj)
                else:
                    stoplisted_objects.append(obj)
                    logger.info(f"Исключен объект с in_stoplist={obj_in_stoplist}: {obj.get('name', 'Без имени')}")
        
        objects = safe_objects
        
        # Обновляем ответ с учетом фильтрации
        if stoplisted_objects:
            answer = f"{answer} (исключено {len(stoplisted_objects)} объектов по уровню безопасности)"
        
        # Debug информация о фильтрации stoplist
        debug_info["stoplist_filter"] = {
            "total_before_filter": len(result.get("objects", [])),
            "safe_after_filter": len(objects),
            "stoplisted_count": len(stoplisted_objects)
        }
        
        if not objects:
            response = {
                "status": "no_objects", 
                "message": answer,
                "used_objects": [],
                "not_used_objects": []
            }
            if debug_mode:
                response["debug"] = debug_info
                response["in_stoplist_filter_applied"] = True
                response["in_stoplist_level"] = in_stoplist
            return jsonify(response)

        # Filter out invalid geometries before visualization
        valid_objects = []
        object_details = []
        
        # ============================================================================
        # ФОРМИРОВАНИЕ used_objects И not_used_objects
        # ============================================================================
        used_objects = []      # Объекты, которые будут на карте
        not_used_objects = []  # Объекты, которые не попали на карту (невалидная геометрия)
        
        for obj in objects:
            try:
                if obj.get("geojson") and obj["geojson"].get("coordinates"):
                    # Basic validation of coordinates
                    if isinstance(obj["geojson"]["coordinates"][0], (int, float)):
                        lon, lat = obj["geojson"]["coordinates"]
                        if -180 <= lon <= 180 and -90 <= lat <= 90:
                            valid_objects.append(obj)
                            object_details.append({
                                "id": obj.get("id", "unknown"),
                                "name": obj.get("name", "Без имени"),
                                "type": obj.get("type", "unknown"),
                                "distance_km": obj.get("distance", "unknown")
                            })
                            # Добавляем в used_objects
                            used_objects.append({
                                "name": obj.get("name", "Без имени"),
                                "type": obj.get("type", "unknown"),
                                "distance_km": obj.get("distance", "unknown"),
                                "geometry_type": "point"
                            })
                    elif isinstance(obj["geojson"]["coordinates"][0], list):
                        # For polygons/multipoints, check first coordinate
                        first_coord = obj["geojson"]["coordinates"][0][0]
                        if isinstance(first_coord, (int, float)):
                            if -180 <= first_coord <= 180:
                                valid_objects.append(obj)
                                object_details.append({
                                    "id": obj.get("id", "unknown"),
                                    "name": obj.get("name", "Без имени"),
                                    "type": obj.get("type", "unknown"),
                                    "distance_km": obj.get("distance", "unknown")
                                })
                                # Добавляем в used_objects
                                used_objects.append({
                                    "name": obj.get("name", "Без имени"),
                                    "type": obj.get("type", "unknown"),
                                    "distance_km": obj.get("distance", "unknown"),
                                    "geometry_type": "polygon"
                                })
                        elif len(first_coord) >= 2:
                            lon, lat = first_coord[:2]
                            if -180 <= lon <= 180 and -90 <= lat <= 90:
                                valid_objects.append(obj)
                                object_details.append({
                                    "id": obj.get("id", "unknown"),
                                    "name": obj.get("name", "Без имени"),
                                    "type": obj.get("type", "unknown"),
                                    "distance_km": obj.get("distance", "unknown")
                                })
                                # Добавляем в used_objects
                                used_objects.append({
                                    "name": obj.get("name", "Без имени"),
                                    "type": obj.get("type", "unknown"),
                                    "distance_km": obj.get("distance", "unknown"),
                                    "geometry_type": "complex"
                                })
                else:
                    # Объект без геометрии - добавляем в not_used_objects
                    not_used_objects.append({
                        "name": obj.get("name", "Без имени"),
                        "type": obj.get("type", "unknown"),
                        "distance_km": obj.get("distance", "unknown"),
                        "reason": "no_geometry"
                    })
            except Exception as e:
                logger.warning(f"Invalid geometry in object {obj.get('name')}: {str(e)}")
                # Добавляем в not_used_objects с причиной ошибки
                not_used_objects.append({
                    "name": obj.get("name", "Без имени"),
                    "type": obj.get("type", "unknown"),
                    "distance_km": obj.get("distance", "unknown"),
                    "reason": "invalid_geometry",
                    "error": str(e)
                })
                continue

        debug_info["valid_objects_count"] = len(valid_objects)
        debug_info["object_details"] = object_details
        debug_info["validation_errors"] = len(objects) - len(valid_objects)

        if not valid_objects:
            response = {
                "status": "error",
                "message": "Найдены объекты, но их координаты недействительны для отображения",
                "used_objects": [],
                "not_used_objects": not_used_objects
            }
            if debug_mode:
                response["debug"] = debug_info
                response["in_stoplist_filter_applied"] = True
                response["in_stoplist_level"] = in_stoplist
            return jsonify(response)

        # 2. Визуализируем только валидные объекты
        try:
            # Создаем карту с именем из redis_key (кешированное имя)
            map_name = redis_key.replace("cache:", "map_").replace(":", "_")
            map_result = geo.draw_custom_geometries(valid_objects, map_name)
            t3 = time.perf_counter()
            map_result["count"] = len(valid_objects)
            map_result["answer"] = answer
            map_result["names"] = [obj.get("name", "Без имени") for obj in valid_objects]
            
            # ДОБАВЛЯЕМ used_objects и not_used_objects К СУЩЕСТВУЮЩЕЙ СТРУКТУРЕ
            map_result["used_objects"] = used_objects
            map_result["not_used_objects"] = not_used_objects
            
            # Добавляем информацию о фильтрации по stoplist
            map_result["in_stoplist_filter_applied"] = True
            map_result["in_stoplist_level"] = in_stoplist
            map_result["stoplisted_count"] = len(stoplisted_objects)
            
            # Добавляем информацию о разрешении синонимов в ответ
            if resolved_species_info and resolved_species_info.get("resolved", False):
                map_result["species_synonym_resolution"] = {
                    "original_name": resolved_species_info["original_name"],
                    "resolved_name": species_name,
                    "resolved": True
                }
            
            # Добавляем debug информацию
            debug_info["render_time"] = round(t3 - t2, 3)
            debug_info["total_time"] = round(time.perf_counter() - t0, 3)
            debug_info["map_generation"] = {
                "static_map": map_result.get("static_map"),
                "interactive_map": map_result.get("interactive_map"),
                "map_name": map_name
            }
            
            if debug_mode:
                map_result["debug"] = debug_info

            # Сохраняем в кеш (30 минут для поиска по координатам)
            set_cached_result(redis_key, map_result, expire_time=1800)
                
            return jsonify(map_result)
        except Exception as e:
            logger.error(f"Ошибка отрисовки карты: {e}")
            debug_info["render_error"] = str(e)
            response = {
                "status": "error", 
                "message": f"Ошибка отрисовки карты: {e}",
                "objects": [obj["name"] for obj in valid_objects],
                "answer": answer,
                "in_stoplist_filter_applied": True,
                "in_stoplist_level": in_stoplist,
                "used_objects": used_objects,
                "not_used_objects": not_used_objects
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 500
            
    except Exception as e:
        logger.error(f"Ошибка поиска рядом: {e}")
        debug_info["search_error"] = str(e)
        response = {
            "status": "error", 
            "message": f"Ошибка поиска рядом: {e}",
            "used_objects": [],
            "not_used_objects": []
        }
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 500
    finally:
        logging.info(
            "coords_to_map timings parse=%.3f search=%.3f render=%.3f total=%.3f",
            t_after_parse - t0,
            t2 - t1,
            t3 - t2,
            time.perf_counter() - t0,
        )

@app.route("/find_species_with_description", methods=["POST"])
def find_species_with_description():
    data = request.get_json()
    name = data.get("name")
    limit = data.get("limit", 1500)
    offset = data.get("offset", 0)
    
    logger.info(f"POST /find_species_with_description - name: {name}, limit: {limit}, offset: {offset}")
    
    if not name:
        return jsonify({
            "status": "error",
            "message": "Параметр 'name' обязателен",
            "used_objects": [],
            "not_used_objects": []
        }), 400
    
    result = slot_val.find_species_with_description(name, limit, offset)
    
    # Добавляем информацию об объектах
    used_objects = []
    not_used_objects = []
    
    if result.get("status") == "success" and result.get("results"):
        # Все найденные виды считаются использованными
        for species in result["results"]:
            used_objects.append({
                "name": species.get("name", name),
                "type": "biological_entity"
            })
    else:
        # Если ничего не найдено, объект попадает в not_used_objects
        not_used_objects.append({
            "name": name,
            "type": "biological_entity" 
        })
    
    # Обновляем результат с объектами
    result["used_objects"] = used_objects
    result["not_used_objects"] = not_used_objects
    
    return jsonify(result)

@app.route("/")
def home():
    return "SalutBot API works!"

# Запускается через gunicorn
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)