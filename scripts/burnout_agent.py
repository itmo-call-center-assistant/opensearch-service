#!/usr/bin/env python3
"""
Агент для анализа риска выгорания и генерации рекомендаций на основе:
- табличных данных по сотруднику,
- предсказаний модели burnout_model_v1.pkl,
- RAG-поиска по корпусу (OpenSearch + YandexGPT, без HyDE).

Использование (из терминала):
    python3 scripts/burnout_agent.py /path/to/burnout_synthetic.csv 1001
где 1001 — employee_id.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
try:  # sklearn может отсутствовать на этапе статического анализа
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # noqa: BLE001
    InconsistentVersionWarning = None  # type: ignore[assignment]

from scripts.services.search_service import SearchService
from scripts.services.local_llm_service import LocalLLMService


FEATURES: List[str] = [
    "load_change",
    "overtime_change",
    "days_since_vacation_norm",
    "was_on_sick_leave",
    "has_reprimand",
    "participates_in_activities",
    "has_subordinates",
    "kpi1",
    "kpi2",
    "kpi3",
    "kpi4",
    "kpi5",
    "age",
    "tenure",
]


@dataclass
class RiskPrediction:
    risk_proba: float
    risk_level: str


class BurnoutRiskService:
    """Обёртка над burnout_model_v1.pkl."""

    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Подавляем предупреждение InconsistentVersionWarning от sklearn,
        # чтобы не засорять логи при загрузке модели из другой версии sklearn.
        if InconsistentVersionWarning is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=InconsistentVersionWarning,
                )
                self.model = joblib.load(model_path)
        else:
            self.model = joblib.load(model_path)

    def predict_row(self, row: pd.Series) -> RiskPrediction:
        x = row[FEATURES].values.reshape(1, -1)
        proba = float(self.model.predict_proba(x)[0, 1])
        if proba < 0.25:
            level = "low"
        elif proba < 0.6:
            level = "medium"
        else:
            level = "high"
        return RiskPrediction(risk_proba=proba, risk_level=level)


class PatternExtractor:
    """
    Формирует профиль сотрудника для LLM без жёстких правил/порогов.

    Задача этого класса — просто собрать числовые признаки в удобный JSON,
    а интерпретацию (какие факторы являются рисками и насколько) оставить на LLM.
    """

    @staticmethod
    def extract(row: pd.Series, risk: RiskPrediction) -> Dict[str, Any]:
        features = {
            "load_change": float(row["load_change"]),
            "overtime_change": float(row["overtime_change"]),
            "days_since_vacation_norm": float(row["days_since_vacation_norm"]),
            "was_on_sick_leave": int(row["was_on_sick_leave"]),
            "has_reprimand": int(row["has_reprimand"]),
            "participates_in_activities": int(row["participates_in_activities"]),
            "has_subordinates": int(row["has_subordinates"]),
            "kpi1": float(row["kpi1"]),
            "kpi2": float(row["kpi2"]),
            "kpi3": float(row["kpi3"]),
            "kpi4": float(row["kpi4"]),
            "kpi5": float(row["kpi5"]),
            "age": int(row["age"]),
            "tenure": float(row["tenure"]),
        }

        kpis = [
            features["kpi1"],
            features["kpi2"],
            features["kpi3"],
            features["kpi4"],
            features["kpi5"],
        ]
        kpi_mean = sum(kpis) / len(kpis)

        return {
            "risk_proba": round(risk.risk_proba, 4),
            "risk_level": risk.risk_level,
            "features": features,
            "kpi_mean": round(float(kpi_mean), 3),
        }


class BurnoutAgent:
    """Агент, объединяющий модель риска и RAG-поиск для выдачи рекомендаций.

    Базовый поток всегда использует YandexGPT (через SearchService.yandex_service).
    Опционально, по желанию пользователя, может дополнительно запрашивать ответ
    у локальной LLM (LocalLLMService). Таким образом:
    - YandexGPT остаётся «истиной по умолчанию»,
    - локальная LLM даёт второй вариант ответа, если настроена.
    """

    def __init__(self, model_path: str, use_local_llm: bool = False) -> None:
        self.risk_service = BurnoutRiskService(model_path)
        self.search_service = SearchService()
        self.use_local_llm = use_local_llm
        self.local_llm_service: Optional[LocalLLMService] = (
            LocalLLMService() if use_local_llm else None
        )

    def _build_rag_query(self, pattern_info: Dict[str, Any]) -> str:
        """Сформировать текст запроса в RAG на основе числовых признаков и риска."""
        risk_level = pattern_info.get("risk_level", "unknown")
        risk_proba = pattern_info.get("risk_proba", 0.0)
        features = pattern_info.get("features", {})

        # Компактное текстовое описание профиля для поискового запроса
        age = features.get("age")
        tenure = features.get("tenure")
        load_change = features.get("load_change")
        overtime_change = features.get("overtime_change")
        days_norm = features.get("days_since_vacation_norm")
        has_subordinates = features.get("has_subordinates")
        participates = features.get("participates_in_activities")
        kpi_mean = pattern_info.get("kpi_mean")

        profile_text = (
            f"возраст {age} лет, стаж {tenure} лет, "
            f"рост нагрузки {load_change:.2f}, рост переработок {overtime_change:.2f}, "
            f"давность отпуска (нормированная) {days_norm:.2f}, "
            f"средний KPI {kpi_mean:.2f}, "
            f"руководитель: {'да' if has_subordinates else 'нет'}, "
            f"участие в активностях: {'есть' if participates else 'нет'}"
        )

        query = (
            "профилактика эмоционального выгорания для сотрудника "
            f"с уровнем риска {risk_level} (вероятность примерно {risk_proba:.0%}), "
            f"учитывая параметры работы: {profile_text}"
        )
        return query

    def _docs_to_context(
        self,
        docs: List[Dict[str, Any]],
        max_chars: int | None = None,
    ) -> str:
        """
        Собрать текстовый контекст из документов RAG.

        Теперь по умолчанию выводим ЧАНКИ ПОЛНОСТЬЮ (без обрезки текста),
        чтобы downstream‑модель могла использовать весь фрагмент.
        max_chars можно задать явно, если нужно ограничить длину.
        """
        parts: List[str] = []
        total = 0
        for i, d in enumerate(docs, 1):
            src = d.get("source", "")
            cid = d.get("chunk_id", "")
            text = (d.get("text") or "").strip()
            block = f"[Фрагмент {i} | source={src} | chunk_id={cid}]\n{text}\n"

            if max_chars is not None:
                total += len(block)
                if total > max_chars:
                    break

            parts.append(block)

        return "\n".join(parts)

    def _build_advice_prompt(
        self,
        row: pd.Series,
        pattern_info: Dict[str, Any],
        docs: List[Dict[str, Any]],
    ) -> str:
        """
        Построить промпт для YandexGPT.

        LLM полностью формирует гипотезы о рисках и рекомендации,
        мы только передаём сырые данные и фрагменты базы знаний.
        """
        features = pattern_info.get("features", {})
        employee_profile = {
            "features": features,
            "kpi_mean": pattern_info.get("kpi_mean"),
            "risk_proba": pattern_info.get("risk_proba"),
            "risk_level": pattern_info.get("risk_level"),
        }

        context = self._docs_to_context(docs)

        system_prompt = (
            "Ты — корпоративный психолог и эксперт по профилактике эмоционального выгорания. "
            "Тебе переданы сырые числовые данные о рабочей нагрузке, отдыхе, KPI и роли сотрудника, "
            "оценка риска выгорания от модели, а также фрагменты из профессиональных материалов "
            "по профилактике выгорания (база знаний). "
            "Твоя задача — на основе ЭТИХ ФРАГМЕНТОВ и данных о сотруднике сформировать цельное, "
            "поддерживающее обращение к нему. "
            "Особенно опирайся на идеи и рекомендации, которые явно присутствуют во фрагментах из базы знаний; "
            "не придумывай методики, которых там нет, и не давай медицинских диагнозов. "
            "Пиши по-русски, простым человеческим языком, как заботливый коллега или друг."
        )

        user_prompt = f"""
ДАННЫЕ И ОЦЕНКА МОДЕЛИ (JSON, для справки):
{json.dumps(employee_profile, ensure_ascii=False, indent=2)}

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ (фрагменты, на которые нужно опираться в рекомендациях):
{context}

ЗАДАЧА:
На основе этих данных и фрагментов напиши одно цельное, тёплое обращение к сотруднику.
Объясни, почему его состояние может быть уязвимым сейчас, какие факторы риска можно предположить
и какие шаги, опираясь на материалы базы знаний, помогут ему снизить напряжение и позаботиться о себе.
Не используй номера пунктов и технические термины, говори живым человеческим языком.
"""

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        return full_prompt

    async def advise_for_row(self, row: pd.Series, top_k_docs: int = 8) -> Dict[str, Any]:
        """Основной метод агента: посчитать риск, сходить в RAG и выдать советы."""
        # 1. Предсказание риска
        risk = self.risk_service.predict_row(row)

        # 2. Паттерны риска
        pattern_info = PatternExtractor.extract(row, risk)

        # 3. Запрос в RAG (без HyDE)
        query_text = self._build_rag_query(pattern_info)
        docs = await self.search_service.search_documents(
            query=query_text,
            size=top_k_docs,
            semantic_weight=0.7,
            keyword_weight=0.3,
            use_hyde=False,
            use_colbert=True,
        )

        # 4. Генерация рекомендаций (YandexGPT + опционально локальная LLM)
        prompt = self._build_advice_prompt(row, pattern_info, docs)
        answer_yandex = await self.search_service.yandex_service.get_completion(
            prompt,
            max_tokens=900,
        )

        answer_local: Optional[str] = None
        if self.use_local_llm and self.local_llm_service is not None:
            answer_local = await self.local_llm_service.get_completion(
                prompt,
                max_tokens=900,
            )

        return {
            "risk": pattern_info,
            "rag_query": query_text,
            # Для обратной совместимости оставляем поле `answer` как ответ YandexGPT
            "answer": answer_yandex,
            "answer_yandex": answer_yandex,
            "answer_local": answer_local,
            "docs": docs,
        }

    async def advise_for_features(
        self,
        features_dict: Dict[str, Any],
        top_k_docs: int = 8,
        index_name: str | None = None,
        use_hyde: bool = False,
        use_colbert: bool = True,
    ) -> Dict[str, Any]:
        """
        Вариант основного метода агента для уже подготовленных признаков (без CSV).

        На вход подаются числовые/категориальные признаки сотрудника в виде dict.
        LLM сама формирует гипотезы и рекомендации, мы лишь подготавливаем данные
        и RAG‑контекст.
        """
        row = pd.Series(features_dict)

        # 1. Предсказание риска
        risk = self.risk_service.predict_row(row)

        # 2. Паттерны риска
        pattern_info = PatternExtractor.extract(row, risk)

        # 3. Запрос в RAG
        query_text = self._build_rag_query(pattern_info)
        docs = await self.search_service.search_documents(
            query=query_text,
            size=top_k_docs,
            semantic_weight=0.7,
            keyword_weight=0.3,
            use_hyde=use_hyde,
            use_colbert=use_colbert,
            index_name=index_name,
        )

        # 4. Генерация рекомендаций (YandexGPT + опционально локальная LLM)
        prompt = self._build_advice_prompt(row, pattern_info, docs)
        answer_yandex = await self.search_service.yandex_service.get_completion(
            prompt,
            max_tokens=900,
        )

        answer_local: Optional[str] = None
        if self.use_local_llm and self.local_llm_service is not None:
            answer_local = await self.local_llm_service.get_completion(
                prompt,
                max_tokens=900,
            )

        return {
            "risk": pattern_info,
            "rag_query": query_text,
            "answer": answer_yandex,
            "answer_yandex": answer_yandex,
            "answer_local": answer_local,
            "docs": docs,
        }


def _load_row_from_csv(csv_path: str, employee_id: Optional[int]) -> Tuple[pd.Series, int]:
    df = pd.read_csv(csv_path)
    if employee_id is not None and "employee_id" in df.columns:
        row = df.loc[df["employee_id"] == employee_id]
        if row.empty:
            raise ValueError(f"employee_id={employee_id} not found in {csv_path}")
        idx = int(row.index[0])
        return row.iloc[0], idx
    # Если employee_id не указан — берём первую строку
    return df.iloc[0], int(df.index[0])


async def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Burnout RAG агент: анализ риска и рекомендации.",
    )
    parser.add_argument(
        "csv_path",
        help="Путь к CSV с данными сотрудников (burnout_synthetic.csv или sdek_burnout_real_input.csv).",
    )
    parser.add_argument(
        "--employee-id",
        type=int,
        default=None,
        help="employee_id нужного сотрудника (если не указан — берётся первая строка).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "burnout_model_v1.pkl"),
        help="Путь к burnout_model_v1.pkl (по умолчанию ищется в корне проекта).",
    )
    parser.add_argument(
        "--use-local-llm",
        action="store_true",
        help=(
            "Если указан — помимо YandexGPT дополнительно запрашивать ответ у "
            "локальной LLM (LOCAL_LLM_URL)."
        ),
    )
    args = parser.parse_args()

    row, idx = _load_row_from_csv(args.csv_path, args.employee_id)

    agent = BurnoutAgent(model_path=args.model_path, use_local_llm=args.use_local_llm)
    result = await agent.advise_for_row(row)

    # Консольный вывод
    print("=== EMPLOYEE ROW INDEX ===")
    print(idx)
    if "employee_id" in row:
        print(f"employee_id: {int(row['employee_id'])}")
    print("\n=== RISK SUMMARY ===")
    print(json.dumps(result["risk"], ensure_ascii=False, indent=2))

    print("\n=== RAG QUERY ===")
    print(result["rag_query"])

    print("\n=== ADVICE (YandexGPT) ===")
    print(result.get("answer_yandex") or result.get("answer"))

    if result.get("answer_local"):
        print("\n=== ADVICE (Local LLM) ===")
        print(result["answer_local"])

    print("\n=== USED DOCS (top 5) ===")
    for i, d in enumerate(result["docs"][:5], 1):
        print(
            f"[{i}] source={d.get('source')} chunk_id={d.get('chunk_id')} "
            f"score={d.get('_score')} colbert={d.get('_colbert_score')}",
        )

    out_payload = {
        "employee_row_index": idx,
        "employee_id": int(row["employee_id"]) if "employee_id" in row else None,
        "risk": result["risk"],
        "rag_query": result["rag_query"],
        "answer": result["answer"],
        "docs_top5": result["docs"][:5],
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "burnout_agent_last.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_cli_main())


