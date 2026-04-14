# search_api/services/llm_answer_generator.py
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from ..domain.entities import ObjectResult, ResourceResult
from .llm_integration import get_llm
import logging

logger = logging.getLogger(__name__)


class LLMAnswerGenerator:
    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def generate(self, question: str, objects: List[ObjectResult], resources: List[ResourceResult]) -> Dict[str, Any]:
        context = self._build_context(objects, resources)
        llm = self._get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Ты эксперт по Байкальской природной территории. "
                "Используй твою базу знаний для точных ответов на вопросы пользователя.\n\n"
                "Особые указания:\n"
                "- На вопросы 'сколько' - подсчитай количество соответствующих записей в базе знаний\n"
                "Например, на вопрос 'Сколько музеев?' при информации 'Всего найдено записей: 98 (в контекст включено топ-5 по релевантности)', нужно ответить около 98 музеев и затем описание каждого музея из топ записей\n"
                "- Будь информативным и лаконичным\n"
                "- Начинай ответ с прямого ответа на запрос пользователя, отвечай ТОЛЬКО на него\n"
                "- При запросе 'Какие другие достопримечательности есть?' нужно описать месторождения из твоей базы и другие достопримечательности которые ты знаешь!\n"
                "- Даже при неполной информации предоставь доступные детали\n\n"
                "Твоя база знаний:\n{context}\n\n"
                "Вопрос: {question}\n\n"
                "Ответ:"
            ))
        ])
        try:
            chain = prompt | llm
            response = chain.invoke({"question": question, "context": context})
            content = response.content.strip() if hasattr(response, 'content') else str(response)
            finish_reason = None
            if hasattr(response, 'response_metadata'):
                finish_reason = response.response_metadata.get('finish_reason')
            if not finish_reason and hasattr(response, 'additional_kwargs'):
                finish_reason = response.additional_kwargs.get('finish_reason')
            is_success = bool(content)
            if finish_reason == 'blacklist':
                is_success = False
            return {
                "content": content,
                "finish_reason": finish_reason,
                "success": is_success
            }
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {
                "content": "Извините, не удалось сгенерировать ответ на основе доступной информации.",
                "finish_reason": "error",
                "success": False
            }

    def _build_context(self, objects: List[ObjectResult], resources: List[ResourceResult]) -> str:
        context_parts = []
        if objects:
            context_parts.append(f"Найдено объектов: {len(objects)}")
            for obj in objects[:10]:
                props_str = str(obj.properties)[:200] if obj.properties else "{}"
                context_parts.append(f"- Объект: {obj.object_type} '{obj.db_id}': свойства: {props_str}")
        if resources:
            context_parts.append(f"Найдено ресурсов: {len(resources)}")
            for res in resources[:10]:
                content_preview = str(res.content)[:200] if res.content else "Нет данных"
                context_parts.append(f"- Ресурс: {res.title} (тип: {res.modality_type}): {content_preview}")
        return "\n".join(context_parts) if context_parts else "Нет релевантной информации."