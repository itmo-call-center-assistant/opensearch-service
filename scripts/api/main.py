from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI

from scripts.burnout_agent import BurnoutAgent, BurnoutRiskService
from scripts.api.schemas import (
    AgentRequest,
    AgentResponse,
    EmployeeFeatures,
    LLMRequest,
    LLMResponse,
    PredictRequest,
    RiskResponse,
    SearchRequest,
    SearchResponse,
    QARequest,
    QAResponse,
)
from scripts.services.search_service import SearchService
from scripts.opensearch_config import OpenSearchConfig


app = FastAPI(
    title="Burnout RAG API",
    description="Сервисы для оценки риска выгорания и RAG-поиска по материалам.",
    version="1.0.0",
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "burnout_model_v1.pkl")
burnout_agent = BurnoutAgent(model_path=MODEL_PATH)
risk_service = burnout_agent.risk_service
search_service = burnout_agent.search_service


def _features_to_row_dict(features: EmployeeFeatures) -> Dict[str, Any]:
    """Вспомогательная функция: Pydantic-модель -> dict для pandas.Series."""
    return features.dict()


@app.post("/predict", response_model=RiskResponse)
async def predict_risk(payload: PredictRequest) -> RiskResponse:
    """
    Предсказание риска эмоционального выгорания по признакам сотрудника.
    """
    row_dict = _features_to_row_dict(payload)
    import pandas as pd

    row = pd.Series(row_dict)
    risk = risk_service.predict_row(row)

    return RiskResponse(
        risk_proba=risk.risk_proba,
        risk_level=risk.risk_level,
        details={
            "features": row_dict,
        },
    )


@app.post("/agent/advise", response_model=AgentResponse)
async def agent_advise(payload: AgentRequest) -> AgentResponse:
    """
    RAG-агент: считает риск, делает поиск по базе знаний и возвращает
    поддерживающее текстовое обращение к сотруднику.
    """
    features_dict = _features_to_row_dict(payload)
    result = await burnout_agent.advise_for_features(
        features_dict=features_dict,
        top_k_docs=payload.top_k_docs,
        index_name=payload.index_name,
        use_hyde=payload.use_hyde,
        use_colbert=payload.use_colbert,
    )
    return AgentResponse(
        risk=result["risk"],
        rag_query=result["rag_query"],
        answer=result["answer"],
        docs=result["docs"],
    )


@app.post("/search", response_model=SearchResponse)
async def search_documents(payload: SearchRequest) -> SearchResponse:
    """
    Чистый поиск по OpenSearch с опциональными HyDE и ColBERT.
    """
    documents = await search_service.search_documents(
        query=payload.query,
        size=payload.size,
        semantic_weight=payload.semantic_weight if hasattr(payload, "semantic_weight") else 0.7,
        keyword_weight=payload.keyword_weight if hasattr(payload, "keyword_weight") else 0.3,
        use_hyde=payload.use_hyde,
        use_colbert=payload.use_colbert,
        index_name=payload.index_name,
    )
    return SearchResponse(
        query=payload.query,
        total_documents=len(documents),
        documents=documents,
    )


@app.post("/rag/answer", response_model=QAResponse)
async def rag_answer(payload: QARequest) -> QAResponse:
    """
    RAG-эндпоинт: принимает вопрос, выполняет поиск по выбранному индексу
    и генерирует ответ по найденным документам.
    """
    result = await search_service.search_and_answer(
        query=payload.query,
        size=payload.size,
        semantic_weight=0.7,
        keyword_weight=0.3,
        use_hyde=payload.use_hyde,
        use_colbert=payload.use_colbert,
        index_name=payload.index_name,
    )
    return QAResponse(
        query=result["query"],
        answer=result["answer"],
        total_documents=result["total_documents"],
        documents=result["documents"],
    )


@app.post("/llm/answer", response_model=LLMResponse)
async def llm_answer(payload: LLMRequest) -> LLMResponse:
    """
    Запрос к LLM.

    Если index_name не задан — это прямой вызов модели без RAG.
    Если index_name указан — перед ответом выполняется поиск по выбранному индексу
    и ответ строится на основе найденных документов (RAG-режим).

    Таким образом, этот эндпоинт можно использовать и как "чистый" чат,
    и как чат-психолога по базе документов.
    """
    if payload.index_name:
        documents = await search_service.search_documents(
            query=payload.query,
            size=payload.size,
            semantic_weight=0.7,
            keyword_weight=0.3,
            use_hyde=payload.use_hyde,
            use_colbert=payload.use_colbert,
            index_name=payload.index_name,
        )
        answer = await search_service.generate_answer(
            query=payload.query,
            context_documents=documents,
            max_tokens=payload.max_tokens,
        )
    else:
    answer = await search_service.yandex_service.get_completion(
        prompt=payload.query,
        max_tokens=payload.max_tokens,
    )

    return LLMResponse(
        query=payload.query,
        answer=answer,
    )




