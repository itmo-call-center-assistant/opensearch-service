"""Сервис для обращения к локальной LLM (через HTTP API).

Идея простая:
- во ВСЕХ основных потоках по‑прежнему используется YandexGPT (эмбеддинги + основная генерация),
- локальная LLM — это дополнительный источник ответа «по желанию пользователя».

Интеграция управляется переменными окружения:
- LOCAL_LLM_URL        — обязательный URL эндпоинта (например, OpenAI‑совместимый /v1/chat/completions),
- LOCAL_LLM_MODEL      — имя модели (по умолчанию: \"local-model\"),
- LOCAL_LLM_API_KEY    — опциональный токен (если нужен для эндпоинта).

Формат запроса предполагается OpenAI‑совместимый (messages[]), чтобы легко
подцепить Ollama/vLLM/локальный прокси. При необходимости формат можно
адаптировать под конкретный сервер.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx


logger = logging.getLogger(__name__)


class LocalLLMService:
    """Обёртка над локальной LLM.

    Если LOCAL_LLM_URL не задан, сервис считается выключенным и
    возвращает аккуратные заглушки.
    """

    def __init__(self) -> None:
        self.url: Optional[str] = os.getenv("LOCAL_LLM_URL")
        self.model: str = os.getenv("LOCAL_LLM_MODEL", "local-model")
        self.api_key: Optional[str] = os.getenv("LOCAL_LLM_API_KEY")

        self.enabled: bool = bool(self.url)
        if not self.enabled:
            logger.info(
                "LocalLLMService: LOCAL_LLM_URL не задан — локальная LLM отключена.",
            )

    async def get_completion(self, prompt: str, max_tokens: int = 1000) -> str:
        """Получить ответ от локальной LLM.

        По умолчанию предполагаем OpenAI‑совместимый чат‑эндпоинт.
        При необходимости формат можно изменить под конкретный сервер.
        """
        if not self.enabled or not self.url:
            return (
                "Локальная LLM сейчас не настроена (переменная LOCAL_LLM_URL не задана). "
                "Продолжаю работать только через YandexGPT."
            )

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
                resp = await client.post(self.url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.error("LocalLLMService: ошибка при запросе к локальной LLM: %s", e)
            return (
                "Не удалось получить ответ от локальной LLM. "
                "Используйте ответ от YandexGPT."
            )

        try:
            # OpenAI‑совместимый формат: choices[0].message.content
            choices = data.get("choices") or []
            if not choices:
                return "Локальная LLM вернула пустой ответ. Используйте ответ от YandexGPT."
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            return str(content)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "LocalLLMService: ошибка парсинга ответа локальной LLM: %s", e
            )
            return (
                "Не удалось корректно разобрать ответ локальной LLM. "
                "Используйте ответ от YandexGPT."
            )



