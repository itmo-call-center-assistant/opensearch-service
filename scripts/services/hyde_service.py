"""
HyDE (Hypothetical Document Embeddings) — вспомогательный сервис для усиления поиска.

Адаптация файла из проекта «2гис» без изменений логики,
использует Yandex GPT для генерации гипотетических документов.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


class HyDEProcessor:
    """Класс для реализации HyDE‑подхода."""

    def __init__(self) -> None:
        self.yandex_api_key = os.getenv("YANDEX_API_KEY")
        self.yandex_folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.yandex_llm_model = os.getenv("YANDEX_LLM_MODEL", "yandexgpt-lite")
        self.yandex_embedding_model = os.getenv(
            "YANDEX_EMBEDDING_MODEL", "text-search-doc"
        )

        self.headers = {
            "Authorization": f"Api-Key {self.yandex_api_key or ''}",
            "x-folder-id": self.yandex_folder_id or "",
            "Content-Type": "application/json",
        }

    async def generate_hypothetical_documents(
        self, query: str, num_hypotheses: int = 1
    ) -> List[str]:
        """Генерирует гипотетические документы для запроса."""
        try:
            print(f"HyDE: генерируем гипотезы для запроса: '{query}'")
            hypotheses: List[str] = []

            for i in range(num_hypotheses):
                prompts = [
                    f"Ключевые слова для поиска: {query}",
                    f"Поисковые термины: {query}",
                    f"Что искать: {query}",
                    f"Поиск: {query}",
                ]

                prompt = prompts[i % len(prompts)]
                print(f"HyDE: используем промпт {i + 1}: {prompt[:100]}...")
                hypothesis = await self._generate_hypothesis(prompt)

                if hypothesis and hypothesis not in hypotheses:
                    hypotheses.append(hypothesis)
                    print(
                        f"HyDE: сгенерирована гипотеза {i + 1}: {hypothesis[:100]}..."
                    )
                else:
                    print(f"HyDE: гипотеза {i + 1} не сгенерирована или дублируется")

            print(f"HyDE: всего сгенерировано гипотез: {len(hypotheses)}")
            return hypotheses
        except Exception as e:  # noqa: BLE001
            logger.error("Ошибка при генерации гипотетических документов: %s", e)
            return []

    async def _generate_hypothesis(self, prompt: str) -> Optional[str]:
        """Генерирует одну гипотезу через Yandex GPT."""
        try:
            data = {
                "modelUri": f"gpt://{self.yandex_folder_id}/{self.yandex_llm_model}",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.7,
                    "maxTokens": 100,
                },
                "messages": [
                    {
                        "role": "system",
                        "text": (
                            "Ты эксперт по поиску. Генерируй только короткие ключевые "
                            "слова и термины для поиска, не более 10–15 слов. "
                            "Не создавай длинные описания."
                        ),
                    },
                    {
                        "role": "user",
                        "text": prompt,
                    },
                ],
            }

            async with httpx.AsyncClient(verify=False, timeout=30) as client:
                response = await client.post(
                    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                    headers=self.headers,
                    json=data,
                )

            print(f"HyDE API ответ: {response.status_code}")
            if response.status_code != 200:
                print(f"HyDE API ошибка: {response.text}")
                return None

            result = response.json()
            alternatives = result.get("result", {}).get("alternatives", [])
            if not alternatives:
                print("HyDE: нет alternatives в ответе")
                return None

            hypothesis = alternatives[0].get("message", {}).get("text", "")
            if not hypothesis:
                print("HyDE: пустая гипотеза в ответе")
                return None

            print(f"HyDE гипотеза сгенерирована: {hypothesis[:100]}...")
            return hypothesis.strip()
        except Exception as e:  # noqa: BLE001
            logger.error("Ошибка при генерации гипотезы: %s", e)
            return None


# Глобальный экземпляр, как в оригинальном проекте
hyde_processor = HyDEProcessor()


