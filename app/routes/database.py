import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.config import BASE_DIR, RESOURCES_DIST_PATH, IMAGES_DIR
from app.services import relational_service
from core.resource_update_service import ResourceUpdateService

database_bp = Blueprint('database', __name__)
logger = logging.getLogger(__name__)

@database_bp.route("/reload_database", methods=["POST"])
def reload_database():
    """
    Эндпоинт для перезагрузки базы данных без добавления новых ресурсов
    (параметр use_stubs удалён, так как эмбеддинги больше не используются в реляционной БД)
    """
    try:
        logger.info(f"📤 /reload_database - получен запрос")
        
        reload_database_param = request.form.get('reload_database', 'false').lower() == 'true'
        incremental = request.form.get('incremental', 'true').lower() == 'true'
        
        logger.info(f"Параметры запроса:")
        logger.info(f"  - reload_database: {reload_database_param}")
        logger.info(f"  - incremental: {incremental}")
        
        if not reload_database_param:
            return jsonify({
                "status": "error",
                "message": "Параметр reload_database должен быть true для перезагрузки БД",
                "used_objects": [],
                "not_used_objects": []
            }), 400
        
        if not os.path.exists(RESOURCES_DIST_PATH):
            logger.error(f"Файл resources_dist.json не найден: {RESOURCES_DIST_PATH}")
            return jsonify({
                "status": "error",
                "message": f"Файл resources_dist.json не найден",
                "used_objects": [],
                "not_used_objects": []
            }), 404
        
        service = ResourceUpdateService(RESOURCES_DIST_PATH, IMAGES_DIR)
        
        # Вызываем метод без use_stubs
        results = service.reload_database_only(
            reload_database=reload_database_param,
            incremental=incremental
        )
        
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

@database_bp.route("/upload_resources", methods=["POST"])
def upload_resources():
    """
    Эндпоинт для загрузки архивов с аннотациями и изображениями
    (параметр use_stubs удалён)
    """
    try:
        logger.info(f"📤 /upload_resources - получен запрос")
        
        has_json = 'json_archive' in request.files and request.files['json_archive'].filename
        has_images = 'images_archive' in request.files and request.files['images_archive'].filename
        
        reload_database = request.form.get('reload_database', 'false').lower() == 'true'
        incremental_update = request.form.get('incremental_update', 'true').lower() == 'true'
        
        logger.info(f"Параметры запроса:")
        logger.info(f"  - Есть JSON архив: {has_json}")
        logger.info(f"  - Есть архив изображений: {has_images}")
        logger.info(f"  - reload_database: {reload_database}")
        logger.info(f"  - incremental_update: {incremental_update}")
        
        if not os.path.exists(RESOURCES_DIST_PATH):
            logger.error(f"Файл resources_dist.json не найден: {RESOURCES_DIST_PATH}")
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
        
        temp_dir = tempfile.mkdtemp()
        json_archive_path = None
        images_archive_path = None
        
        try:
            if has_json:
                json_file = request.files['json_archive']
                filename = secure_filename(json_file.filename)
                json_archive_path = os.path.join(temp_dir, filename)
                json_file.save(json_archive_path)
                logger.info(f"Сохранен JSON архив: {json_archive_path} ({os.path.getsize(json_archive_path)} байт)")
            
            if has_images:
                images_file = request.files['images_archive']
                filename = secure_filename(images_file.filename)
                images_archive_path = os.path.join(temp_dir, filename)
                images_file.save(images_archive_path)
                logger.info(f"Сохранен архив с изображениями: {images_archive_path} ({os.path.getsize(images_archive_path)} байт)")
            
            service = ResourceUpdateService(RESOURCES_DIST_PATH, IMAGES_DIR)
            
            # Вызываем метод без use_stubs
            results = service.process_upload(
                json_archive_path=json_archive_path,
                images_archive_path=images_archive_path,
                reload_database=reload_database,
                incremental=incremental_update
            )
            
            if reload_database:
                results["update_type"] = "полное" if not incremental_update else "инкрементальное"
            else:
                results["update_type"] = "без обновления БД"
            
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
            
            logger.info(f"✅ Обработка завершена: {results}")
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