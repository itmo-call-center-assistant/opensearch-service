from __future__ import annotations

from typing import List, Dict, Any

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests.auth import HTTPBasicAuth

try:
    from ..opensearch_config import OpenSearchConfig
except ImportError:
    from opensearch_config import OpenSearchConfig


class OpenSearchService:
    def __init__(self, cfg: OpenSearchConfig) -> None:
        self.cfg = cfg
        self.client = self._create_client(cfg)
        self._ensure_search_pipeline()

    def _create_client(self, cfg: OpenSearchConfig) -> OpenSearch:
        auth = None
        if cfg.user and cfg.password:
            auth = HTTPBasicAuth(cfg.user, cfg.password)
        client = OpenSearch(
            hosts=[cfg.url],
            http_compress=True,
            http_auth=auth,
            use_ssl=cfg.url.startswith("https://"),
            verify_certs=False,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
        return client

    def _ensure_search_pipeline(self) -> None:
        """
        Если в конфиге задан search-pipeline, подключаем его как
        index.search.default_pipeline для индекса по умолчанию.

        Это аналог кода:
            es.indices.put_settings(
                index=index,
                body={\"index.search.default_pipeline\": \"nlp-test-pipeline\"}
            )
        """
        if not self.cfg.search_pipeline:
            return
        try:
            self.client.indices.put_settings(
                index=self.cfg.index_name,
                body={
                    "index.search.default_pipeline": self.cfg.search_pipeline,
                },
            )
            print(
                "OpenSearchService: set index.search.default_pipeline="
                f"{self.cfg.search_pipeline!r} for index {self.cfg.index_name!r}",
            )
        except Exception as e:
            print(
                "OpenSearchService: failed to set search.default_pipeline "
                f"for index {self.cfg.index_name!r}: {e}",
            )

    def search(
        self,
        query: str,
        k: int = 8,
        index_name: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Повторяет логику 2гис: основной BM25 запрос + windowed rescore на top-N.

        По умолчанию используется индекс из cfg.index_name, но можно переопределить
        через параметр index_name (например, для выбора корпуса на уровне API).
        """
        body: Dict[str, Any] = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text^2", "source"],
                    "type": "best_fields",
                    "operator": "or",
                }
            },
            "highlight": {"fields": {"text": {}}},
            "rescore": {
                "window_size": max(k, 50),
                "query": {
                    "rescore_query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["text^2", "source"],
                            "type": "phrase",
                            "slop": 3,
                        }
                    },
                    "query_weight": 1.0,
                    "rescore_query_weight": 1.2,
                },
            },
        }
        target_index = index_name or self.cfg.index_name
        res = self.client.search(index=target_index, body=body)
        hits = res.get("hits", {}).get("hits", [])
        out: List[Dict[str, Any]] = []
        for h in hits:
            src = h.get("_source", {})
            src["_score"] = h.get("_score")
            src["highlight"] = (h.get("highlight") or {}).get("text", [])
            out.append(src)
        return out


