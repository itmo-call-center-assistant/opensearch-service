"""Основной сервис поиска в стиле проекта «2гис».

Объединяет:
- YandexService (эмбеддинги + LLM),
- OpenSearchService (BM25/Hybrid поиск по индексу с фрагментами текста),
- HyDE (опционально),
- ColBERT‑подобный реранкер.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Импортируем конфиги и сервисы так, чтобы модуль корректно работал
# как часть пакетного модуля `scripts.services.*`, так и в старом сценарии
# запуска из папки `scripts`.
try:  # вариант для пакетного импорта: scripts.services.search_service
    from scripts.opensearch_config import OpenSearchConfig
    from scripts.services.opensearch_service import OpenSearchService
    from scripts.services.yandex_service import YandexService
    from scripts.services.hyde_service import hyde_processor
    from scripts.services.colbert_reranker import colbert_reranker
except ImportError:  # вариант для локального запуска из каталога `scripts`
    from opensearch_config import OpenSearchConfig
    from services.opensearch_service import OpenSearchService
    from services.yandex_service import YandexService
    from services.hyde_service import hyde_processor
    from services.colbert_reranker import colbert_reranker


class SearchService:
    """Основной сервис поиска, объединяющий Yandex GPT и OpenSearch."""

    def __init__(
        self,
        os_cfg: Optional[OpenSearchConfig] = None,
        os_service: Optional[OpenSearchService] = None,
        yandex_service: Optional[YandexService] = None,
    ) -> None:
        self.os_cfg = os_cfg or OpenSearchConfig()
        self.yandex_service = yandex_service or YandexService()
        self.opensearch_service = os_service or OpenSearchService(self.os_cfg)

    async def search_documents(
        self,
        query: str,
        size: int = 10,
        semantic_weight: float = 0.7,  # для совместимости с 2гис; пока не используется
        keyword_weight: float = 0.3,  # для совместимости с 2гис; пока не используется
        use_hyde: bool = True,
        use_colbert: bool = True,
        index_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Поиск документов с HyDE, гибридным (BM25 + kNN) поиском в OpenSearch и ColBERT‑реранкингом.

        index_name позволяет явно задать индекс OpenSearch (корпус),
        если он отличается от значения по умолчанию в OpenSearchConfig.
        """
        try:
            # 1. Базовый embedding запроса
            query_embedding = await self.yandex_service.get_embedding(query)

            # 2. HyDE (опционально)
            if use_hyde:
                query_embedding = await self._apply_hyde(query, query_embedding)

            # 3. Гибридный поиск в OpenSearch: kNN по text_vector + keyword (multi_match)
            #    Если гибрид недоступен/даёт ошибку — откатываемся на старый BM25‑поиск.
            raw_size = size * 2 if use_colbert else size
            target_index = index_name or self.os_cfg.index_name
            client = self.opensearch_service.client

            try:
                body: Dict[str, Any] = {
                    "size": raw_size,
                    "query": {
                        "hybrid": {
                            "queries": [
                                {
                                    "knn": {
                                        "text_vector": {
                                            "vector": query_embedding,
                                            "k": raw_size,
                                        }
                                    }
                                },
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["text^2", "source"],
                                        "type": "best_fields",
                                        # Важно: используем OR, чтобы длинные запросы
                                        # не «зажимали» выдачу.
                                        "operator": "or",
                                    }
                                },
                            ]
                        }
                    },
                    "_source": ["text", "source", "chunk_id"],
                }
                res = client.search(index=target_index, body=body)
                hits = res.get("hits", {}).get("hits", [])
                results: List[Dict[str, Any]] = []
                for h in hits:
                    src = h.get("_source", {}) or {}
                    src["_score"] = h.get("_score")
                    # подсветку не настраиваем в этом запросе, оставляем пустой список
                    src["highlight"] = []
                    results.append(src)
                print(
                    f"Hybrid search (BM25 + kNN) вернул {len(results)} документов "
                    f"из индекса {target_index}",
                )
            except Exception as e:  # noqa: BLE001
                # Фолбэк на прежний BM25‑поиск, если гибрид по какой‑то причине не поддерживается
                print(f"Гибридный поиск не удался, откатываемся на BM25: {e}")
                results = self.opensearch_service.search(
                    query,
                    k=raw_size,
                    index_name=index_name,
                )

            # 4. ColBERT‑реранкинг (опционально)
            print(
                f"ColBERT включен: {use_colbert}, результатов: "
                f"{len(results) if results else 0}",
            )
            if use_colbert and results:
                print(f"Применяем ColBERT реранкинг для {len(results)} результатов")
                results = await colbert_reranker.rerank_results(
                    query,
                    results,
                    top_k=size,
                )
                print(f"После ColBERT реранкинга: {len(results)} результатов")
            elif use_colbert:
                print("ColBERT реранкинг пропущен — нет результатов")
            else:
                print("ColBERT реранкинг отключен")

            return results
        except Exception as e:  # noqa: BLE001
            print(f"Ошибка при поиске документов: {e}")
            return []

    async def _apply_hyde(
        self,
        query: str,
        original_embedding: List[float],
    ) -> List[float]:
        """Применить HyDE для улучшения embedding запроса."""
        try:
            hypotheses = await hyde_processor.generate_hypothetical_documents(
                query,
                num_hypotheses=2,
            )
            if not hypotheses:
                return original_embedding

            hypothesis_embeddings = await hyde_processor.create_embeddings(hypotheses)
            combined_embedding = hyde_processor.combine_embeddings(
                original_embedding,
                hypothesis_embeddings,
            )
            print(f"HyDE применён: сгенерировано гипотез: {len(hypotheses)}")
            return combined_embedding
        except Exception as e:  # noqa: BLE001
            print(f"Ошибка при применении HyDE: {e}")
            return original_embedding

    async def generate_answer(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        max_tokens: int = 800,
    ) -> str:
        """Сгенерировать ответ по найденным фрагментам из индекса."""
        try:
            context_parts: List[str] = []
            for i, doc in enumerate(context_documents, 1):
                src = doc.get("source", "")
                cid = doc.get("chunk_id", "")
                text = (doc.get("text") or "").strip()
                context_parts.append(
                    f"Фрагмент {i} (source={src}, chunk_id={cid}):\n{text}\n",
                )

            context = "\n".join(context_parts)

            prompt = f"""
Ты — эксперт‑помощник. Отвечай по приведённым ниже фрагментам текста.
Если информации недостаточно, честно скажи, чего не хватает.
Отвечай по‑русски, кратко и по делу, без выдумок.

Вопрос пользователя:
{query}

Контекст:
{context}
"""
            return await self.yandex_service.get_completion(prompt, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"Ошибка при генерации ответа: {e}")
            return "Извините, произошла ошибка при генерации ответа."

    async def search_and_answer(
        self,
        query: str,
        size: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        use_hyde: bool = True,
        use_colbert: bool = True,
        index_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Полный цикл: поиск документов + генерация ответа (как в 2гис)."""
        try:
            print(
                f"Поиск с параметрами: HyDE={use_hyde}, ColBERT={use_colbert}, "
                f"index={index_name or self.os_cfg.index_name}",
            )
            documents = await self.search_documents(
                query=query,
                size=size,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                use_hyde=use_hyde,
                use_colbert=use_colbert,
                index_name=index_name,
            )
            answer = await self.generate_answer(query, documents)
            return {
                "query": query,
                "answer": answer,
                "documents": documents,
                "total_documents": len(documents),
            }
        except Exception as e:  # noqa: BLE001
            print(f"Ошибка при поиске и генерации ответа: {e}")
            return {
                "query": query,
                "answer": "Извините, произошла ошибка при обработке запроса.",
                "documents": [],
                "total_documents": 0,
            }



