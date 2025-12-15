"""Сервис для работы с Yandex GPT API (эмбеддинги + генерация текста).

Адаптация под структуру из проекта «2гис», но конфигурация берётся
из переменных окружения, как в текущем проекте.
"""

from __future__ import annotations

import logging
import os
from typing import List

import httpx


logger = logging.getLogger(__name__)


class YandexService:
    """Сервис для работы с Yandex GPT / Foundation Models.

    ВАЖНО: если переменные окружения для Yandex не заданы, сервис не падает,
    а возвращает безопасные заглушки, чтобы всё API продолжало работать.
    """

    def __init__(self) -> None:
        self.api_key: str | None = os.getenv("YANDEX_API_KEY")
        self.folder_id: str | None = os.getenv("YANDEX_FOLDER_ID")

        self.embedding_url: str = os.getenv(
            "YANDEX_EMBEDDINGS_URL",
            "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
        )
        self.embedding_model: str = os.getenv(
            "YANDEX_EMBEDDING_MODEL",
            "text-search-doc",
        )

        self.completion_url: str = os.getenv(
            "YANDEX_COMPLETION_URL",
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        )
        self.llm_model: str = os.getenv("YANDEX_LLM_MODEL", "yandexgpt-lite")

        self.enabled: bool = bool(self.api_key and self.folder_id)
        if not self.enabled:
            logger.warning(
                "YandexService: YANDEX_API_KEY / YANDEX_FOLDER_ID не заданы — "
                "будут использованы заглушки вместо реального Yandex GPT.",
            )

    async def get_embedding(self, text: str) -> List[float]:
        """Получить векторное представление текста через Yandex Foundation Models."""
        if not self.enabled:
            return [0.0] * 256

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "modelUri": f"emb://{self.folder_id}/{self.embedding_model}",
            "text": text,
        }

        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            resp = await client.post(
                self.embedding_url,
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        if "result" in data and "embedding" in data["result"]:
            return data["result"]["embedding"]
        if "embedding" in data:
            return data["embedding"]

        return [0.0] * 256

    async def get_completion(self, prompt: str, max_tokens: int = 1000) -> str:
        """Получить ответ от Yandex GPT."""
        if not self.enabled:
            logger.info(
                "YandexService: запрос completion без настроенного API‑ключа — "
                "возвращаем заглушку.",
            )
            return (
                "Сервис генерации текста сейчас не настроен (нет Yandex API ключа). "
                "Обратитесь к администратору, чтобы добавить YANDEX_API_KEY и "
                "YANDEX_FOLDER_ID в переменные окружения."
            )

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.llm_model}",
            "completionOptions": {
                "maxTokens": max_tokens,
                "temperature": 0.1,
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt,
                }
            ],
        }

        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            resp = await client.post(
                self.completion_url,
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        return (
            data.get("result", {})
            .get("alternatives", [{}])[0]
            .get("message", {})
            .get("text", "")
        )
