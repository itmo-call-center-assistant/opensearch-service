"""
Быстрый ColBERT‑подобный реранкер на основе document‑level embeddings.

Адаптация файла из проекта «2гис» под нашу схему документов:
используем поле `text` как основной корпус документа.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class FastColBERTReranker:
    """Быстрый реранкер на основе document‑level embeddings (Yandex FM)."""

    def __init__(self) -> None:
        self.yandex_embedding_model = os.getenv(
            "YANDEX_EMBEDDING_MODEL",
            "text-search-doc",
        )

        logger.info("ColBERT: Embedding модель: %s", self.yandex_embedding_model)
        self.yandex_api_key: Optional[str] = None
        self.yandex_folder_id: Optional[str] = None
        self.headers: Dict[str, str] = {}
        self._refresh_credentials()

    def _refresh_credentials(self) -> None:
        """
        Актуализировать ключи и заголовки из переменных окружения.

        Важно для ноутбуков, где YANDEX_API_KEY / YANDEX_FOLDER_ID
        могут выставляться уже после импорта модуля.
        """
        self.yandex_api_key = os.getenv("YANDEX_API_KEY")
        self.yandex_folder_id = os.getenv("YANDEX_FOLDER_ID")

        if not self.yandex_api_key or not self.yandex_folder_id:
            logger.warning(
                "ColBERT: YANDEX_API_KEY / YANDEX_FOLDER_ID не заданы — "
                "реранкинг будет работать только на BM25 без семантики",
            )
        self.headers = {
            "Authorization": f"Api-Key {self.yandex_api_key or ''}",
            "x-folder-id": self.yandex_folder_id or "",
            "Content-Type": "application/json",
        }

    async def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Реранжировать результаты поиска по ColBERT‑score."""
        try:
            if not results:
                print("ColBERT: нет результатов для реранкинга")
                return []

            print(f"ColBERT: начинаем реранжирование {len(results)} результатов")

            query_embedding = await self._get_embedding(query)
            if not query_embedding:
                print("ColBERT: не удалось получить embedding для запроса")
                return results[:top_k]

            reranked = await self._fast_colbert_rerank(query_embedding, results)
            return reranked[:top_k]
        except Exception as e:
            print(f"ColBERT: ошибка при реранжировании: {e}")
            return results[:top_k]

    async def _fast_colbert_rerank(
        self,
        query_embedding: List[float],
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Быстрое реранжирование с использованием document‑level embeddings."""
        try:
            self._refresh_credentials()
            if not self.yandex_api_key or not self.yandex_folder_id:
                logger.info(
                    "ColBERT: реранкинг пропущен — нет Yandex‑креденшелов",
                )
                return results

            print("ColBERT: применяем реранжирование")
            scored_results: List[Dict[str, Any]] = []

            for i, result in enumerate(results):
                text = (result.get("text") or "").strip()

                if not text:
                    colbert_score = 0.0
                else:
                    doc_embedding = await self._get_embedding(text)
                    if doc_embedding:
                        colbert_score = self._cosine_similarity(
                            query_embedding, doc_embedding
                        )
                        print(
                            f"ColBERT: документ {i + 1}, embedding получен, "
                            f"score={colbert_score:.3f}",
                        )
                    else:
                        colbert_score = 0.0
                        print(f"ColBERT: документ {i + 1}, embedding не получен")

                original_score = float(result.get("_score", 0.0) or 0.0)
                combined_score = 0.8 * colbert_score + 0.2 * self._normalize_score(
                    original_score
                )

                result_copy = result.copy()
                result_copy["_rerank_score"] = combined_score
                result_copy["_colbert_score"] = colbert_score
                scored_results.append(result_copy)

            reranked = sorted(
                scored_results,
                key=lambda x: x.get("_rerank_score", 0.0),
                reverse=True,
            )
            return reranked
        except Exception as e:
            print(f"ColBERT: ошибка в реранжировании: {e}")
            return results

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Получить embedding для текста через Yandex API."""
        try:
            self._refresh_credentials()
            if not self.yandex_api_key or not self.yandex_folder_id:
                print(
                    "ColBERT: пропускаем embedding — не установлены "
                    "YANDEX_API_KEY / YANDEX_FOLDER_ID",
                )
                return None

            cleaned_text = " ".join(text.split())
            if not cleaned_text:
                print("ColBERT: пустой текст для embedding")
                return None

            data = {
                "modelUri": f"emb://{self.yandex_folder_id}/{self.yandex_embedding_model}",
                "text": cleaned_text,
            }

            async with httpx.AsyncClient(verify=False, timeout=30) as client:
                response = await client.post(
                    "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
                    headers=self.headers,
                    json=data,
                )

            print(f"ColBERT: API ответ {response.status_code}")
            if response.status_code != 200:
                print(f"ColBERT: API ошибка {response.status_code}: {response.text}")
                return None

            result = response.json()
            embedding = result.get("embedding") or result.get("result", {}).get(
                "embedding",
            )
            if embedding:
                print(f"ColBERT: получен embedding длиной {len(embedding)}")
                return embedding

            print("ColBERT: пустой embedding в ответе")
            return None
        except Exception as e:
            print(f"ColBERT: ошибка при получении embedding: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Косинусное сходство между двумя векторами."""
        try:
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return 0.0

            v1 = np.asarray(vec1, dtype=np.float32)
            v2 = np.asarray(vec2, dtype=np.float32)
            dot = float(np.dot(v1, v2))
            norm1 = float(np.linalg.norm(v1))
            norm2 = float(np.linalg.norm(v2))
            if norm1 == 0.0 or norm2 == 0.0:
                return 0.0
            sim = dot / (norm1 * norm2)
            return max(0.0, min(1.0, (sim + 1.0) / 2.0))
        except Exception as e:
            print(f"ColBERT: ошибка при расчёте косинусного сходства: {e}")
            return 0.0

    def _normalize_score(self, score: float) -> float:
        """Нормализовать BM25‑скор в диапазон [0, 1]."""
        try:
            if score <= 0:
                return 0.0
            if score >= 10:
                return 1.0
            return score / 10.0
        except Exception as e:
            print(f"ColBERT: ошибка при нормализации скора: {e}")
            return 0.5


colbert_reranker = FastColBERTReranker()


