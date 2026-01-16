from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EmployeeFeatures(BaseModel):
    """Признаки, на которых обучена модель риска выгорания."""

    load_change: float = Field(..., description="Изменение нагрузки")
    overtime_change: float = Field(..., description="Изменение переработок")
    days_since_vacation_norm: float = Field(
        ..., description="Нормированное время с последнего отпуска"
    )
    was_on_sick_leave: int = Field(..., description="Был ли на больничном (0/1)")
    has_reprimand: int = Field(..., description="Есть ли взыскания (0/1)")
    participates_in_activities: int = Field(
        ..., description="Участвует ли в активностях (0/1)"
    )
    has_subordinates: int = Field(..., description="Есть ли подчинённые (0/1)")
    kpi1: float
    kpi2: float
    kpi3: float
    kpi4: float
    kpi5: float
    age: int
    tenure: float


class PredictRequest(EmployeeFeatures):
    """Запрос на предсказание риска выгорания."""


class RiskResponse(BaseModel):
    """Ответ с риском выгорания."""

    risk_proba: float
    risk_level: str
    details: Dict[str, Any]


class AgentRequest(EmployeeFeatures):
    """Запрос к агенту: признаки сотрудника + параметры поиска."""

    top_k_docs: int = Field(8, description="Сколько документов брать после реранкинга")
    index_name: Optional[str] = Field(
        None,
        description="Имя индекса OpenSearch; если не задано — используется значение по умолчанию",
    )
    use_hyde: bool = Field(
        False, description="Использовать ли HyDE при формировании эмбеддинга запроса"
    )
    use_colbert: bool = Field(True, description="Включать ли ColBERT‑реранкер")


class AgentResponse(BaseModel):
    """Ответ агента: риск + текст рекомендаций + использованные документы."""

    risk: Dict[str, Any]
    rag_query: str
    answer: str
    docs: List[Dict[str, Any]]


class SearchRequest(BaseModel):
    """Чистый поиск по индексу OpenSearch."""

    query: str
    size: int = Field(10, description="Сколько документов вернуть")
    index_name: Optional[str] = Field(
        None,
        description="Имя индекса OpenSearch; если не задано — используется значение по умолчанию",
    )
    use_hyde: bool = Field(False, description="Использовать ли HyDE")
    use_colbert: bool = Field(True, description="Использовать ли ColBERT‑реранкинг")


class SearchResponse(BaseModel):
    """Ответ на поиск."""

    query: str
    total_documents: int
    documents: List[Dict[str, Any]]


class QARequest(SearchRequest):
    """RAG-вопрос: поиск + генерация ответа."""

    max_tokens: int = Field(800, description="Максимальное число токенов в ответе LLM")


class QAResponse(BaseModel):
    """Ответ RAG-сервиса."""

    query: str
    answer: str
    total_documents: int
    documents: List[Dict[str, Any]]


class LLMRequest(BaseModel):
    """Запрос к LLM.

    По умолчанию — просто вызов модели без контекста.
    Если указан index_name, используется RAG-контекст по выбранному индексу.
    """

    query: str
    max_tokens: int = Field(800, description="Максимальное число токенов в ответе LLM")
    index_name: Optional[str] = Field(
        None,
        description=(
            "Имя индекса OpenSearch. "
            "Если задано — перед ответом LLM выполняется поиск по этому индексу."
        ),
    )
    size: int = Field(
        5,
        description="Сколько документов взять из индекса для RAG-контекста (если index_name задан).",
    )
    use_hyde: bool = Field(
        False,
        description="Использовать ли HyDE при формировании эмбеддинга запроса (для RAG-режима).",
    )
    use_colbert: bool = Field(
        True,
        description="Включать ли ColBERT‑реранкер (для RAG-режима).",
    )


class LLMResponse(BaseModel):
    """Ответ LLM без контекста документов."""

    query: str
    answer: str
