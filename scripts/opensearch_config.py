import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class OpenSearchConfig:
    """
    Базовая конфигурация для подключения к OpenSearch.

    ВАЖНО:
    - Хост и порт задаются через переменные окружения:
      OPENSEARCH_HOST, OPENSEARCH_PORT
    - Подключение всегда идёт по HTTPS.
    """

    host: str = os.getenv("OPENSEARCH_HOST", "localhost")
    port: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    user: Optional[str] = os.getenv("OPENSEARCH_USER")
    password: Optional[str] = os.getenv("OPENSEARCH_PASSWORD")
    index_name: str = os.getenv("OPENSEARCH_INDEX", "makar_sdek1")
    search_pipeline: Optional[str] = os.getenv("OPENSEARCH_SEARCH_PIPELINE")

    @property
    def url(self) -> str:
        """
        Полный HTTPS‑URL до OpenSearch, который используется в клиентах.
        Например: https://localhost:9200
        """
        return f"https://{self.host}:{self.port}"


@dataclass
class YandexConfig:
    provider: str = os.getenv("LLM_PROVIDER", "yandex")
    api_key: Optional[str] = os.getenv("YANDEX_API_KEY")
    folder_id: Optional[str] = os.getenv("YANDEX_FOLDER_ID")
    model: str = os.getenv("YANDEX_LLM_MODEL", "yandexgpt-lite")
    embedding_model: str = os.getenv("YANDEX_EMBEDDING_MODEL", "text-search-doc")
    completion_url: str = os.getenv(
        "YANDEX_COMPLETION_URL",
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
    )

    def model_uri(self) -> str:
        if not self.folder_id:
            raise ValueError("YANDEX_FOLDER_ID is required")
        return f"gpt://{self.folder_id}/{self.model}"


