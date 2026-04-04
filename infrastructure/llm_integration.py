import os
import logging
import hashlib
from typing import Any, Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_gigachat import GigaChat

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_llm_instances: Dict[str, BaseChatModel] = {}

def get_llm(provider: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> BaseChatModel:
    """
    Возвращает экземпляр LLM в соответствии с выбранным провайдером.

    Args:
        provider: 'gigachat' или 'qwen'. Если не указан, берётся из env LLM_PROVIDER.
        params: Дополнительные параметры для конструктора LLM.

    Returns:
        Экземпляр BaseChatModel (совместим с LangChain).
    """
    global _llm_instances

    provider = provider or os.getenv("LLM_PROVIDER", "gigachat").lower()
    params = params or {}

    # Параметры по умолчанию для каждого провайдера
    if provider == "gigachat":
        default_params = {
            'credentials': os.getenv("GIGACHAT_CREDENTIALS"),
            'model': os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max"),
            'verify_ssl_certs': False,
            'profanity_check': False,
            'temperature': 0.0,
            'timeout': 120,
            'scope': 'GIGACHAT_API_PERS'
        }
        final_params = {**default_params, **params}
        # Ключ кэша
        cache_key = hashlib.md5(
            f"{provider}_{str(sorted(final_params.items()))}".encode()
        ).hexdigest()

        if cache_key not in _llm_instances:
            logger.info(f"Создаём новый экземпляр GigaChat с параметрами: {final_params}")
            _llm_instances[cache_key] = GigaChat(**final_params)
        return _llm_instances[cache_key]

    elif provider == "qwen":
        default_params = {
            'base_url': os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            'api_key': os.getenv("LLM_API_KEY", "ollama"),
            'model': os.getenv("LLM_MODEL", "qwen2.5:14b"),
            'temperature': 0.1,
        }
        final_params = {**default_params, **params}
        cache_key = hashlib.md5(
            f"{provider}_{str(sorted(final_params.items()))}".encode()
        ).hexdigest()

        if cache_key not in _llm_instances:
            logger.info(f"Создаём новый экземпляр Qwen (ChatOpenAI) с параметрами: {final_params}")
            _llm_instances[cache_key] = ChatOpenAI(**final_params)
        return _llm_instances[cache_key]

    else:
        raise ValueError(f"Неподдерживаемый провайдер LLM: {provider}. Используйте 'gigachat' или 'qwen'.")