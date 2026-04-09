# search_api/services/llm_integration.py
import os
import logging
import hashlib
from typing import Any, Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

try:
    from langchain_gigachat import GigaChat
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False
    GigaChat = None

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_llm_instances: Dict[str, BaseChatModel] = {}

def get_llm(provider: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> BaseChatModel:
    provider = provider or os.getenv("LLM_PROVIDER", "gigachat").lower()
    params = params or {}

    if provider == "gigachat":
        if not GIGACHAT_AVAILABLE:
            logger.warning("GigaChat not available, falling back to qwen")
            provider = "qwen"
        else:
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
            cache_key = hashlib.md5(
                f"{provider}_{str(sorted(final_params.items()))}".encode()
            ).hexdigest()
            if cache_key not in _llm_instances:
                logger.info(f"Creating GigaChat instance with params: {final_params}")
                _llm_instances[cache_key] = GigaChat(**final_params)
            return _llm_instances[cache_key]

    if provider == "qwen":
        default_params = {
            'base_url': os.getenv("LLM_BASE_URL", "http://100.122.155.85:11434/v1"),
            'api_key': os.getenv("LLM_API_KEY", "ollama"),
            'model': os.getenv("LLM_MODEL", "qwen2.5:14b"),
            'temperature': 0.1,
        }
        final_params = {**default_params, **params}
        cache_key = hashlib.md5(
            f"{provider}_{str(sorted(final_params.items()))}".encode()
        ).hexdigest()
        if cache_key not in _llm_instances:
            logger.info(f"Creating Qwen instance with params: {final_params}")
            _llm_instances[cache_key] = ChatOpenAI(**final_params)
        return _llm_instances[cache_key]

    raise ValueError(f"Unsupported LLM provider: {provider}. Use 'gigachat' or 'qwen'.")