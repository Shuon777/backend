import logging
from flask import Blueprint, request, jsonify
from app.services import relational_service

error_log_bp = Blueprint('error_log', __name__)
logger = logging.getLogger(__name__)

@error_log_bp.route("/log_error", methods=["POST"])
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
        
        logger.info(f"📝 Логирование ошибки: {error_message}...")
        
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