## Обзор

Этот проект — демонстрация RAG‑системы для оценки риска профессионального выгорания и генерации персональных рекомендаций.  
Стек:

- **OpenSearch** — хранилище базы знаний и гибридный поиск (BM25 + kNN + ColBERT‑реранкинг).
- **Yandex Foundation Models** — эмбеддинги (`text-search-doc`) и генерация текста (YandexGPT).
- **FastAPI** — HTTP‑API с несколькими эндпоинтами (`/predict`, `/agent/advise`, `/search`, `/rag/answer`, `/llm/answer`).
- **Sklearn‑модель** `burnout_model_v1.pkl` — предсказание риска выгорания.

Ниже — пошаговая инструкция, как:

1. Настроить окружение и переменные.
2. Проиндексировать базу знаний в OpenSearch.
3. Запустить API на порту `8000`.
4. Делать запросы ко всем эндпоинтам.

---

## 1. Предварительные условия

- **Python / Conda**: используется окружение Anaconda (у вас уже есть `/Users/admin/anaconda3`).
- **OpenSearch**:
  - Должен быть запущен локально и доступен по `https://localhost:9201`.
  - Логин/пароль администратора, например:
    - логин: `admin`
    - пароль: `Toshi545454!`
- **Файлы проекта** (лежат в корне `/Users/admin/СДЭК`):
  - `burnout_model_v1.pkl` — модель риска выгорания.
  - `opensearch_index_makar_sdek1.ipynb` — ноутбук для индексации в OpenSearch.
  - `corpus.json` или другие исходные материалы для базы знаний (PDF/тексты, которые уже используются в ноутбуке).

### 1.1. Установка зависимостей (если нужно)

Если проект разворачивается в новом окружении, установите основные пакеты:

```bash
cd /Users/admin/СДЭК

pip install fastapi uvicorn[standard] opensearch-py httpx pandas scikit-learn pydantic python-dotenv
```

*(У вас большая часть этого уже установлена через Anaconda, команда нужна только при новом разворачивании.)*

---

## 2. Переменные окружения

Проект использует переменные окружения для подключения к OpenSearch и Yandex API.

### 2.1. Настройка переменных окружения

Создайте файл `.env` в корне проекта на основе примера `env.example`:

```bash
cp env.example .env
```

Затем отредактируйте `.env` и укажите ваши реальные значения:

```bash
# OpenSearch Configuration
OPENSEARCH_URL=https://localhost:9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=your_actual_password

# Yandex API Configuration
YANDEX_API_KEY=your_actual_api_key
YANDEX_FOLDER_ID=your_actual_folder_id
```

**Важно:** Файл `.env` не должен попадать в систему контроля версий (добавьте его в `.gitignore`).

### 2.2. Загрузка переменных окружения

Для загрузки переменных из `.env` файла в ноутбуках Jupyter используйте библиотеку `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
```

Или установите переменные окружения вручную перед запуском ноутбука:

```bash
export OPENSEARCH_URL="https://localhost:9200"
export OPENSEARCH_USER="admin"
export OPENSEARCH_PASSWORD="your_password"
export YANDEX_API_KEY="your_api_key"
export YANDEX_FOLDER_ID="your_folder_id"
```

Пример экспорта (вставляйте в терминал **без реальных ключей** — подставьте свои значения):

```bash
cd /Users/admin/СДЭК

export OPENSEARCH_URL="https://localhost:9201"
export OPENSEARCH_USER="admin"
export OPENSEARCH_PASSWORD="ВАШ_ПАРОЛЬ"
export OPENSEARCH_INDEX="makar_sdek1"

export YANDEX_API_KEY="ВАШ_YANDEX_API_KEY"
export YANDEX_FOLDER_ID="ВАШ_FOLDER_ID"
export YANDEX_LLM_MODEL="yandexgpt-lite"
export YANDEX_COMPLETION_URL="https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
export YANDEX_EMBEDDING_MODEL="text-search-doc"
export YANDEX_EMBEDDINGS_URL="https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
```

> В `zsh` не забывайте, что `!` в паролях надо либо экранировать (`\!`), либо оборачивать пароль в одинарные кавычки.

---

## 3. Индексация базы знаний в OpenSearch

Индексация реализована в ноутбуке `opensearch_index_makar_sdek1.ipynb`.  
Он:

- создаёт индекс `makar_sdek1` с кастомными анализаторами (как в 2ГИС),
- настраивает `knn_vector`‑поле для эмбеддингов (`text_vector`, размер 256, HNSW+FAISS),
- получает эмбеддинги текстовых фрагментов через Yandex Embeddings API,
- индексирует документы в OpenSearch.

### 3.1. Логика создания индекса

Индекс создаётся с параметрами, аналогичными:

- **settings.index**:
  - `number_of_shards: 1`
  - `number_of_replicas: 0`
  - `knn: true`
  - `knn.algo_param.ef_search: 100`
  - `similarity.custom_similarity: { type: "BM25", k1: 1.2, b: 0.75 }`
- **analysis**:
  - фильтры: `russian_stemmer`, `unique_pos`, `synonym_graph_filter`, `my_multiplexer` и т.д.
  - анализаторы:
    - `text_analyzer`, `search_text_analyzer`, `ru_international_translit_analyzer`, `exact_analyzer` и др.
- **mappings.properties**:
  - `text: { type: "text", analyzer: "text_analyzer", similarity: "BM25" }`
  - `source: { type: "keyword" }`
  - `chunk_id: { type: "keyword" }`
  - `text_vector: { type: "knn_vector", dimension: 256, space_type: "cosinesimil", ... }`

Все эти шаги уже есть в ноутбуке, вам остаётся только прогнать его.

### 3.2. Как запустить индексацию

1. Убедитесь, что OpenSearch запущен и переменные окружения выставлены (как в разделе 2).
2. Откройте ноутбук:

   ```bash
   cd /Users/admin/СДЭК
   jupyter notebook opensearch_index_makar_sdek1.ipynb
   ```

3. В интерфейсе Jupyter:
   - выберите ядро (то же окружение, где установлен `opensearch-py`, `httpx`, `pandas`),
   - последовательно выполните все ячейки сверху вниз:
     - создание индекса `makar_sdek1`,
     - загрузка корпуса (PDF/JSON),
     - получение эмбеддингов через Yandex,
     - батч‑индексация документов в OpenSearch.
4. В конце вы должны увидеть сообщения о количестве проиндексированных документов.  
   Проверить, что индекс живой, можно будет через эндпоинт `/search` (см. ниже).

---

## 4. Запуск FastAPI‑сервиса на порту 8000

Приложение находится в `scripts/api/main.py` и поднимается через `uvicorn`.

### 4.1. Команда запуска

```bash
cd /Users/admin/СДЭК

export OPENSEARCH_URL="https://localhost:9201"
export OPENSEARCH_USER="admin"
export OPENSEARCH_PASSWORD="ВАШ_ПАРОЛЬ"
export OPENSEARCH_INDEX="makar_sdek1"

export YANDEX_API_KEY="ВАШ_YANDEX_API_KEY"
export YANDEX_FOLDER_ID="ВАШ_FOLDER_ID"
export YANDEX_LLM_MODEL="yandexgpt-lite"
export YANDEX_COMPLETION_URL="https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
export YANDEX_EMBEDDING_MODEL="text-search-doc"
export YANDEX_EMBEDDINGS_URL="https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"

uvicorn scripts.api.main:app --host 0.0.0.0 --port 8000
```

Либо, если хотите использовать конкретный интерпретатор Anaconda:

```bash
/Users/admin/anaconda3/bin/uvicorn scripts.api.main:app --host 0.0.0.0 --port 8000
```

После старта API будет доступен по адресу:

- Swagger UI: `http://127.0.0.1:8000/docs`

---

## 5. Форматы запросов ко всем эндпоинтам

Все эндпоинты — `POST` с JSON‑телом.

### 5.1. `/predict` — предсказание риска выгорания

- **URL**: `POST http://127.0.0.1:8000/predict`
- **Тело запроса** (`PredictRequest` = `EmployeeFeatures`):

```json
{
  "load_change": 0.5,
  "overtime_change": 0.3,
  "days_since_vacation_norm": 0.7,
  "was_on_sick_leave": 0,
  "has_reprimand": 0,
  "participates_in_activities": 1,
  "has_subordinates": 0,
  "kpi1": 0.8,
  "kpi2": 0.7,
  "kpi3": 0.9,
  "kpi4": 0.6,
  "kpi5": 0.75,
  "age": 32,
  "tenure": 3.5
}
```

- **Ответ** (`RiskResponse`):

```json
{
  "risk_proba": 0.1422,
  "risk_level": "low",
  "details": {
    "features": {
      "...": "исходные признаки"
    }
  }
}
```

### 5.2. `/agent/advise` — RAG‑агент (риск + поиск + рекомендации)

- **URL**: `POST http://127.0.0.1:8000/agent/advise`
- **Тело запроса** (`AgentRequest` = `EmployeeFeatures` + параметры поиска):

```json
{
  "load_change": 0.5,
  "overtime_change": 0.3,
  "days_since_vacation_norm": 0.7,
  "was_on_sick_leave": 0,
  "has_reprimand": 0,
  "participates_in_activities": 1,
  "has_subordinates": 0,
  "kpi1": 0.8,
  "kpi2": 0.7,
  "kpi3": 0.9,
  "kpi4": 0.6,
  "kpi5": 0.75,
  "age": 32,
  "tenure": 3.5,

  "top_k_docs": 5,
  "index_name": "makar_sdek1",
  "use_hyde": false,
  "use_colbert": true
}
```

- **Ответ** (`AgentResponse`):

```json
{
  "risk": {
    "risk_proba": 0.1422,
    "risk_level": "low",
    "features": { "...": "сырые признаки" },
    "kpi_mean": 0.75
  },
  "rag_query": "строка поискового запроса, сгенерированная по признакам",
  "answer": "тёплое обращение к сотруднику с рекомендациями",
  "docs": [
    {
      "text": "...фрагмент из сборника...",
      "source": "Sbornik_profilaktika_emotsionalnogo_vygoraniya.pdf",
      "chunk_id": "Sbornik_profilaktika_emotsionalnogo_vygoraniya.pdf::2",
      "_score": 6.29,
      "_colbert_score": 0.82
    }
  ]
}
```

### 5.3. `/search` — чистый поиск по OpenSearch

- **URL**: `POST http://127.0.0.1:8000/search`
- **Тело запроса** (`SearchRequest`):

```json
{
  "query": "профессиональное выгорание профилактика",
  "size": 5,
  "index_name": "makar_sdek1",
  "use_hyde": false,
  "use_colbert": true
}
```

- **Ответ** (`SearchResponse`):

```json
{
  "query": "профессиональное выгорание профилактика",
  "total_documents": 5,
  "documents": [
    {
      "text": "...",
      "source": "Sbornik_profilaktika_emotsionalnogo_vygoraniya.pdf",
      "chunk_id": "Sbornik_profilaktika_emotsionalnogo_vygoraniya.pdf::2",
      "_score": 6.29,
      "_colbert_score": 0.83
    }
  ]
}
```

### 5.4. `/rag/answer` — RAG‑ответ (поиск + генерация)

- **URL**: `POST http://127.0.0.1:8000/rag/answer`
- **Тело запроса** (`QARequest` = `SearchRequest` + `max_tokens`):

```json
{
  "query": "Как можно предотвращать эмоциональное выгорание на работе?",
  "size": 5,
  "index_name": "makar_sdek1",
  "use_hyde": false,
  "use_colbert": true,
  "max_tokens": 600
}
```

- **Ответ** (`QAResponse`):

```json
{
  "query": "Как можно предотвращать эмоциональное выгорание на работе?",
  "answer": "сгенерированный по фрагментам текст с рекомендациями",
  "total_documents": 5,
  "documents": [ { "...": "структура как в /search" } ]
}
```

### 5.5. `/llm/answer` — прямой запрос к LLM (без RAG)

- **URL**: `POST http://127.0.0.1:8000/llm/answer`
- **Тело запроса** (`LLMRequest`):

```json
{
  "query": "Кратко и по-русски объясни, что такое эмоциональное выгорание.",
  "max_tokens": 300
}
```

- **Ответ** (`LLMResponse`):

```json
{
  "query": "Кратко и по-русски объясни, что такое эмоциональное выгорание.",
  "answer": "краткое объяснение от YandexGPT"
}
```

---

## 6. Примеры запросов через `curl`

Для быстрого теста можно использовать `curl`.

### 6.1. Пример `/predict`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "load_change": 0.5,
    "overtime_change": 0.3,
    "days_since_vacation_norm": 0.7,
    "was_on_sick_leave": 0,
    "has_reprimand": 0,
    "participates_in_activities": 1,
    "has_subordinates": 0,
    "kpi1": 0.8,
    "kpi2": 0.7,
    "kpi3": 0.9,
    "kpi4": 0.6,
    "kpi5": 0.75,
    "age": 32,
    "tenure": 3.5
  }'
```

Аналогично можно отправлять JSON для остальных эндпоинтов, подставляя нужные тела запросов из раздела 5.

---

## 7. Что делать дальше

- Чтобы использовать **реальные данные из CSV** (`sdek_burnout_real_input (1).csv`), можно:
  - либо подготовить небольшую утилиту, которая берёт строку по `employee_id` и делает запрос на `/agent/advise`,
  - либо скопировать признаки конкретной строки и вручную подставить их в JSON тела запросов.
- Для экспериментов с промптом агента и поиском удобно комбинировать:
  - `/predict` → посмотреть риск,
  - `/search` → посмотреть, какие фрагменты достаются из OpenSearch,
  - `/agent/advise` → получить полный ответ агента по данным сотрудника.


