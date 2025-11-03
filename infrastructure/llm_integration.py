import os
import logging
from langchain_gigachat import GigaChat
from typing import Any, Dict
import hashlib

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Кэш экземпляров по параметрам
_llm_instances = {}

def get_gigachat(params: Dict[str, Any] = None) -> GigaChat:
    """Возвращает экземпляр GigaChat с заданными параметрами.
    
    Args:
        params: Словарь параметров для конструктора GigaChat
        
    Returns:
        Экземпляр GigaChat
    """
    global _llm_instances
    
    # Параметры по умолчанию
    default_params = {
        'credentials': os.getenv("GIGACHAT_CREDENTIALS"),
        'model': 'GigaChat:latest',
        'verify_ssl_certs': False,
        'profanity_check': False,
        'temperature': 0.1, 
        'timeout': 60
    }
    
    # Объединяем с переданными параметрами
    params = {**default_params, **(params or {})}
    
    # Создаем уникальный ключ для параметров
    params_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()
    
    if params_hash not in _llm_instances:
        try:
            # Создаем новый экземпляр
            _llm_instances[params_hash] = GigaChat(**params)
            logger.info(f"Создан новый экземпляр GigaChat с параметрами: {params}")
        except Exception as e:
            logger.error(f"Ошибка создания GigaChat: {str(e)}")
            # Возвращаем инстанс по умолчанию при ошибке
            return get_gigachat()
    
    return _llm_instances[params_hash]