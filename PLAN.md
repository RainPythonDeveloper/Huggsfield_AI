# PLAN.md — Memory Service for AI Agent (Higgsfield Challenge)

> Итеративный план реализации с метриками на каждом шаге.
> Каждая итерация = одна запись в `CHANGELOG.md` с **What / Why / Result / Next**.
> Дата старта: 2026-05-09. Дедлайн: 2 дня focused work.

---

## 0. Резюме решения (TL;DR)

**Memory Service** — Dockerized HTTP-сервис на **Python 3.12 + FastAPI**, который:
1. Принимает conversation turns через `POST /turns`.
2. Извлекает структурированные memories через LLM (Alem `alemllm`).
3. Хранит их в **Postgres 16 + pgvector + tsvector** (один контейнер с named volume).
4. Отдаёт релевантный контекст через `POST /recall` с гибридным retrieval'ом:
   `(BM25 ⊕ embeddings) → RRF → Alem reranker → priority-aware budget assembly`.
5. Обнаруживает противоречия и поддерживает supersession chains для эволюции фактов.

Контракт §3 из TASK.md выполняется один-в-один. Все 7 endpoint'ов синхронные, eventual consistency исключена.

---

## 1. Tech Stack (зафиксировано)

| Слой | Выбор | Обоснование |
|---|---|---|
| Язык | Python 3.12 | Лучшая экосистема для AI; быстрый prototyping |
| Web | FastAPI + uvicorn | Pydantic-валидация контракта из коробки, async, OpenAPI |
| БД | Postgres 16 + pgvector 0.7 + builtin FTS | Один контейнер: vector + relational + BM25-подобный ranking; persistence через named volume |
| Миграции | `alembic` или ручной `init.sql` (выбор на шаге 1) | MVP-подход: один `init.sql` исполняется при старте контейнера |
| LLM | Alem `alemllm` через OpenAI SDK с `base_url=https://llm.alem.ai/v1` | Endpoint OpenAI-совместимый; JSON wrapped в fences — будем стрипать |
| Embeddings | Alem `text-1024` (dim=1024) | OpenAI-совместимый |
| Reranker | Alem `reranker` через httpx | Cohere-совместимый cross-encoder, сильно разделяет (0.99 vs 0.01 в тесте) |
| HTTP-клиент | `httpx[http2]` + retry | Async, нужен для concurrent extraction calls |
| Tokens | `tiktoken` (cl100k_base) | Для бюджета `max_tokens` в /recall |
| Тесты | `pytest` + `pytest-asyncio` + `httpx.AsyncClient` | Контракт-тесты бьются прямо в FastAPI app |
| Линт/формат | `ruff` (lint+format) | Быстро, один инструмент |
| Контейнер | Multi-stage `python:3.12-slim` | ~200MB финальный образ |
| Оркестрация | `docker-compose.yml` с двумя сервисами: `app` + `db` | `db` — `pgvector/pgvector:pg16`, `app` — наш Dockerfile |

---

## 2. Архитектура

```
┌─────────────────── memory-service container (FastAPI, port 8080) ──────────────────────┐
│                                                                                         │
│  HTTP layer (routes)            Domain services                  Storage                │
│  ─────────────────              ─────────────────                ─────────              │
│  POST /turns      ─────►  ExtractionService  ─────►  ┌──────────────────────────┐      │
│  POST /recall     ─────►  RecallService      ─────►  │ Postgres + pgvector      │      │
│  POST /search     ─────►  SearchService      ─────►  │  - turns                 │      │
│  GET  /memories   ─────►  MemoryRepository   ─────►  │  - messages              │      │
│  DELETE /sessions ─────►  CleanupService     ─────►  │  - memories (vector+fts) │      │
│  DELETE /users    ─────►  CleanupService     ─────►  │  - memory_history        │      │
│  GET  /health     ─────►  (db ping)                  └──────────────────────────┘      │
│                                                                                         │
│         ▲                          ▲                                                    │
│         │                          │                                                    │
│  ┌──────┴──────┐           ┌───────┴──────────┐         ┌──────────────────┐           │
│  │ Alem LLM    │           │ Alem Embeddings  │         │ Alem Reranker    │           │
│  │ /chat/...   │           │ /embeddings      │         │ /rerank          │           │
│  └─────────────┘           └──────────────────┘         └──────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ named volume │  pgdata
                   └──────────────┘
```

### Поток `/turns` (write path)
```
1. Receive turn         ─┐
2. Persist turn+messages │  атомарная транзакция (1 commit)
3. Extract facts (LLM)   │  ←—— синхронно, может занимать 5–30s
4. Resolve contradictions│  для каждого факта: ищем active memory с похожим key
5. Insert/supersede      │  pgvector: вставляем эмбеддинг
6. Return 201            ─┘  ← после этого данные сразу доступны в /recall
```

### Поток `/recall` (read path)
```
1. Embed query (1 call)
2. Optional query rewriting (LLM, 1 call) — для multi-hop / разложение составных вопросов
3. Hybrid retrieval:
   a) BM25 over memory.value + tsvector              → top 30
   b) pgvector cosine over memory.embedding          → top 30
   c) RRF fusion (k=60) → top 20
4. Rerank top 20 через Alem /rerank → top 10
5. Bucket по типу: user_facts (active) / session_recent / search_hit
6. Budget assembly с приоритетом:
   stable_user_facts → query_relevant_memories → recent_session_context
7. Format prose `## Known facts ...` + citations[]
```

---

## 3. Схема БД (Postgres 16 + pgvector)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Conversation history ─────────────────────────────────────────────────
CREATE TABLE turns (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    TEXT NOT NULL,
    user_id       TEXT,                              -- NULL = anonymous
    timestamp     TIMESTAMPTZ NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw           JSONB NOT NULL,                    -- весь incoming payload
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX turns_session_idx ON turns(session_id);
CREATE INDEX turns_user_idx    ON turns(user_id);

CREATE TABLE messages (
    id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id   UUID NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    role      TEXT NOT NULL,                          -- user|assistant|tool
    name      TEXT,
    content   TEXT NOT NULL,
    position  INT NOT NULL,                           -- порядок в turn
    -- tsvector для recall по сырому тексту (fallback при пустых memories)
    content_tsv tsvector GENERATED ALWAYS AS
        (to_tsvector('english', content)) STORED
);
CREATE INDEX messages_turn_idx ON messages(turn_id);
CREATE INDEX messages_tsv_idx  ON messages USING GIN(content_tsv);

-- ── Extracted memories (главная таблица) ─────────────────────────────────
CREATE TABLE memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT,                            -- NULL = session-scoped
    session_id      TEXT,                            -- NULL = global для user
    type            TEXT NOT NULL,                   -- fact|preference|opinion|event|relation
    key             TEXT NOT NULL,                   -- canonical key, e.g. "employer", "city"
    value           TEXT NOT NULL,                   -- canonical value: "Notion"
    raw_quote       TEXT,                            -- цитата из turn (provenance)
    confidence      REAL NOT NULL DEFAULT 0.8,
    embedding       vector(1024) NOT NULL,
    value_tsv       tsvector GENERATED ALWAYS AS
                     (to_tsvector('english', key || ' ' || value)) STORED,
    source_turn     UUID REFERENCES turns(id) ON DELETE SET NULL,
    source_session  TEXT,
    supersedes      UUID REFERENCES memories(id) ON DELETE SET NULL,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX memories_user_active_idx ON memories(user_id) WHERE active;
CREATE INDEX memories_session_idx     ON memories(session_id);
CREATE INDEX memories_key_idx         ON memories(user_id, key) WHERE active;
CREATE INDEX memories_tsv_idx         ON memories USING GIN(value_tsv);
CREATE INDEX memories_embedding_idx   ON memories
    USING hnsw (embedding vector_cosine_ops);

-- supersession уже моделируется через memories.supersedes;
-- отдельная memory_history не нужна, история восстанавливается обходом.
```

**Почему такая схема:**
- `memories` — единая таблица, факт + supersession-ссылка → история восстанавливается через рекурсивный CTE.
- `tsvector` сгенерирован на лету (`GENERATED ALWAYS`) → не нужно поддерживать вручную.
- `HNSW` на embedding → O(log n) ANN search, лучше IVF для маленьких корпусов (наш случай).
- `messages` сохраняет сырой текст для fallback retrieval (если extraction что-то упустил).
- `active` partial index → быстрая выборка только живых фактов.

---

## 4. Структура проекта

```
memory-service/
├── README.md                 # архитектура, обоснования, run instructions
├── CHANGELOG.md              # ИСТОРИЯ ИТЕРАЦИЙ (главный артефакт!)
├── PLAN.md                   # этот файл
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── pyproject.toml
├── src/
│   └── memory/
│       ├── __init__.py
│       ├── main.py           # FastAPI app, lifespan, CORS, error handlers
│       ├── config.py         # Pydantic Settings, читает .env
│       ├── schemas.py        # Pydantic in/out models (контракт §3)
│       ├── db.py             # asyncpg pool, lifespan management
│       ├── migrations/
│       │   └── 001_init.sql
│       ├── routes/
│       │   ├── health.py
│       │   ├── turns.py
│       │   ├── recall.py
│       │   ├── search.py
│       │   ├── memories.py
│       │   └── cleanup.py
│       ├── services/
│       │   ├── extraction.py    # LLM extraction pipeline
│       │   ├── recall.py        # hybrid retrieval + budget assembly
│       │   ├── search.py        # /search structured results
│       │   ├── supersession.py  # contradiction detection + chaining
│       │   └── cleanup.py
│       ├── clients/
│       │   ├── llm.py           # Alem chat completions wrapper
│       │   ├── embeddings.py    # Alem embeddings + batching
│       │   └── reranker.py      # Alem rerank wrapper
│       ├── prompts/
│       │   ├── extract.py       # system+user prompts для extraction
│       │   ├── contradiction.py # промпт для resolve контрадикций
│       │   ├── query_rewrite.py # для multi-hop
│       │   └── format_context.py
│       └── util/
│           ├── tokens.py        # tiktoken counter
│           ├── rrf.py           # reciprocal rank fusion
│           └── json_parse.py    # стрипает ```json fences
├── tests/
│   ├── conftest.py
│   ├── test_contract.py        # roundtrip всех endpoint'ов
│   ├── test_persistence.py     # restart-survival
│   ├── test_concurrency.py     # multi-session isolation
│   ├── test_malformed.py       # 4xx, не падаем
│   ├── test_supersession.py    # fact evolution
│   └── test_recall_quality.py  # читает fixtures/, прогоняет, считает recall@k
└── fixtures/
    ├── conv_career.json        # employment evolution: Stripe → Notion
    ├── conv_pets.json          # implicit fact: "walking Biscuit" → has dog Biscuit
    ├── conv_preferences.json   # opinion arc: TS love → frustration → nuanced
    ├── conv_multihop.json      # 2 фактов в разных turns, склеить в /recall
    ├── conv_noise.json         # запросы про topics которых не было
    └── probes.yaml             # query → expected_facts[] для метрик
```

---

## 5. Iterative Steps with Metrics

> Каждый шаг → атомарный commit + запись в CHANGELOG.md.
> После шагов 4, 6, 7, 8, 9 — обязательный прогон `pytest tests/test_recall_quality.py` и фиксация метрик (recall@k, precision, latency).

### 🟢 Step 0: Repo scaffold + health (30 min)
**Делаем:**
- `Dockerfile`, `docker-compose.yml` (app + db, named volume `pgdata`)
- FastAPI приложение с `GET /health` (включая ping в БД)
- `pyproject.toml` с зависимостями
- `.env.example` со всеми тремя ключами Alem
- Стартовый `init.sql` с pgvector extension и схемой из §3

**Метрика:** `docker compose up -d && curl localhost:8080/health` → `{"status":"ok"}`. Restart container — БД на месте.

**CHANGELOG: v0.1 — Boots, schema in place, no logic yet.**

---

### 🟢 Step 1: Persistence layer + POST /turns (raw store) (1ч)
**Делаем:**
- asyncpg pool + transaction wrapper
- `POST /turns` принимает payload, валидирует через Pydantic, сохраняет в `turns` + `messages`
- `DELETE /sessions/{id}` и `DELETE /users/{id}` (cascade-friendly)
- Pydantic schemas строго по §3

**Метрика:** Smoke-curl из §7 TASK.md проходит для `/turns`. `/users/{id}/memories` пока пустой массив.

**CHANGELOG: v0.2 — Raw turn storage works; persistence verified across restart.**

---

### 🟢 Step 2: Naive recall (embeddings only) — baseline (1ч)
**Делаем:**
- `clients/embeddings.py` с batching и retries
- При insert turn — эмбеддить **сырые messages** (по 1 записи на сообщение в `messages`, эмбеддинг кладём в **временную** колонку `messages.embedding` — позже уберём)
- `POST /recall` = embed(query) → cosine top-5 → склеиваем в context
- `POST /search` = то же самое, но возвращаем структурно

**Метрика на fixture:**
- Создаём `fixtures/probes.yaml` (10–15 проб)
- Запускаем `pytest tests/test_recall_quality.py`
- Записываем **baseline recall@5**: ожидаем 0.30–0.45 (vanilla cosine)

**CHANGELOG: v0.3 — Naive embedding recall. Baseline recall@5 = X.XX.**

> 💡 *Это важная отправная точка: дальше каждое улучшение сравниваем с этим числом.*

---

### 🟢 Step 3: LLM extraction pipeline (2ч)
**Делаем:**
- `clients/llm.py` — вызов Alem chat/completions, стрипаем ```json fences, retry с exponential backoff
- `prompts/extract.py` — system prompt:
  ```
  You are a memory extractor. From the conversation excerpt, extract atomic facts about the user.
  Output strict JSON: {"memories":[{"type": "...", "key":"...", "value":"...", "confidence":0.0-1.0, "raw_quote":"..."}]}
  Types: fact|preference|opinion|event|relation.
  Keys must be canonical lowercase snake_case (employer, city, pet_name, dietary_restriction, ...).
  Capture implicit facts ("walking Biscuit" → pet_name=Biscuit).
  Capture corrections ("actually I meant X" → emit fact + flag corrects_previous=true).
  If no extractable facts, return {"memories":[]}.
  ```
- `services/extraction.py` — после persist turn запускаем extraction, эмбеддим каждый факт, инсёртим в `memories` (пока без supersession)
- Удаляем временный `messages.embedding`, переключаем recall на `memories`

**Метрика:**
- `GET /users/user-1/memories` после прогонки smoke test — должны быть структурированные facts (employer, city, pet_name).
- `recall@5` на fixture: ожидаем 0.45–0.60 (jump относительно baseline за счёт того что recall теперь по канонизированному `key value`, а не по сырому шуму).

**CHANGELOG: v0.4 — LLM extraction. Memories now structured. recall@5: X.XX → Y.YY.**

---

### 🟢 Step 4: Hybrid retrieval (BM25 + embeddings + RRF) (1.5ч)
**Делаем:**
- BM25 через Postgres `ts_rank_cd(value_tsv, plainto_tsquery(...))`
- Параллельно: pgvector cosine
- `util/rrf.py`: `score = Σ 1/(k + rank_i)` с k=60
- Fuse top 30 из каждого канала → top 20 финальный

**Метрика:**
- recall@5 на fixture — ожидаем 0.55–0.70
- Особенно ловим прирост на keyword-heavy queries ("dog's name?", "where works?")
- Latency: фиксируем p50/p95 на фикстуре (logging.info)

**CHANGELOG: v0.5 — Hybrid BM25+embed via RRF. recall@5: Y.YY → Z.ZZ. Keyword queries +N%.**

---

### 🟢 Step 5: Reranker stage (1ч)
**Делаем:**
- `clients/reranker.py` — POST к `/rerank` model=`reranker`, top_n=10
- В пайплайне `/recall`: hybrid → top 20 → rerank → top 10
- Score из reranker'а кладём в `citations[].score` (нормализованный 0..1)

**Метрика:**
- recall@5 — ожидаем 0.65–0.80 (Alem reranker в тесте дал 0.99 vs 0.01 spread → сильный сигнал)
- precision@5 должна вырасти — меньше шума

**CHANGELOG: v0.6 — Added Alem reranker. recall@5: Z.ZZ → W.WW. Precision@5 +N%.**

---

### 🟢 Step 6: Supersession / contradiction handling (2ч)

Это «hard problem» №1 в TASK.md и ключевой grading axis.

**Делаем:**
- При insert нового факта → SELECT active memories с тем же `(user_id, key)` (быстро через partial index)
- Если найдено — двухступенчатая проверка:
  1. **Cheap check:** если `key` совпадает и `value` отличается → flag for resolution
  2. **LLM judge:** один вызов с обоими raw_quotes → JSON `{"verdict":"supersede|coexist|noop","reason":"..."}`
  - "supersede" — старый `active=false`, новый `supersedes=old_id`
  - "coexist" — оба активны (например, "I have a dog AND a cat")
  - "noop" — дубликат, ничего не делаем
- При recall фильтруем только `active=true` факты (старые остаются, но в /recall не попадают)
- `GET /memories` возвращает ВСЕ (включая superseded) с цепочкой → ревьюверы видят историю

**Метрика:**
- Новая fixture `conv_career.json`: turn1 "I work at Stripe" → turn2 "Just started at Notion"
- После двух turns:
  - `/recall "Where do I work?"` → "Notion"
  - `/memories` показывает 2 записи: Notion (active), Stripe (active=false, superseded by Notion)
- `recall@5` на полной fixture (включая evolution probes) — ожидаем 0.70–0.85

**CHANGELOG: v0.7 — Supersession chains. Career-change probe: PASS. recall@5: W.WW → V.VV.**

---

### 🟢 Step 7: Multi-hop via query decomposition (1.5ч)

«What city does the user with the dog named Biscuit live in?» → нужны 2 факта.

**Делаем:**
- `prompts/query_rewrite.py`: LLM-разложение complex queries → 1–3 sub-queries
  ```
  System: Given a recall query, output JSON {"sub_queries": ["...", "..."], "is_multi_hop": bool}
  ```
- Если `is_multi_hop=true` → выполняем hybrid+rerank для каждого sub_query, мерджим citations, дедуп по `memory_id`
- Включаем декомпозицию только когда query содержит anaphora ("the user with...", "their", "his/her") или > 1 fact-axis (через простую эвристику + LLM check)

**Метрика:**
- `conv_multihop.json` fixture (5 multi-hop probes)
- recall@5 на multi-hop subset — ожидаем 0.4 → 0.7

**CHANGELOG: v0.8 — Multi-hop decomposition. Multi-hop recall@5: 0.4 → 0.7.**

---

### 🟢 Step 8: Token budget aware context assembly (1.5ч)
**Делаем:**
- `util/tokens.py` — `tiktoken.get_encoding("cl100k_base")` (близко к нашей токенизации; альтернативно — char/4 эвристика)
- Bucket результатов после rerank:
  - `user_facts`: type ∈ {fact, preference} AND active=true AND user_id=q.user_id (без session фильтра)
  - `query_relevant`: всё что вернул rerank, кроме user_facts
  - `recent`: последние 3 messages из текущей session_id (если ничего другого не зашло)
- Greedy assembly с приоритетом, считаем токены **до** добавления, не превышаем `max_tokens` (целимся в 0.95 от лимита, чтобы был запас)
- Format prose:
  ```
  ## Known facts about this user
  - Works at Notion as a PM (updated 2025-03-15; previously at Stripe as engineer)
  - ...

  ## Relevant from recent conversations
  - [2025-03-10] User mentioned X
  ```

**Метрика:**
- Прогон probes с `max_tokens` ∈ {128, 256, 512, 1024}
- recall@k не должен сильно падать на 512+; на 128 — graceful degradation (только user_facts)
- Никогда не превышаем лимит больше чем на 5%

**CHANGELOG: v0.9 — Budget-aware assembly. Tested at 128/256/512/1024 tokens. No overflow >5%.**

---

### 🟢 Step 9: Robustness + cleanup + concurrency (1ч)
**Делаем:**
- Глобальные FastAPI exception handlers: `RequestValidationError → 422`, `JSONDecodeError → 400`, `Unhandled → 500` (без traceback в response)
- Unicode/emoji/oversized payload tests (limit 1MB)
- DELETE endpoints с CASCADE — проверка что вся связка чистится
- pytest для concurrent sessions: 2 потока пишут в разные session_id одновременно, потом recall не должен показывать кросс-сессионный шум
- Auth middleware: `MEMORY_AUTH_TOKEN` env → если задан, проверяем `Authorization: Bearer`; если не задан — игнорим заголовок

**Метрика:**
- 100% контракт-тестов зелёные
- Стресс: 50 concurrent /turns → не падаем, БД не блокируется
- Restart mid-write (kill -9 во время extraction) → следующий /turns работает; в `memories` не остаётся orphan записей

**CHANGELOG: v1.0-rc — Robustness pass. All contract tests green. Cleanup verified.**

---

### 🟢 Step 10: Final tuning + README + CHANGELOG cleanup (1ч)
**Делаем:**
- Запускаем full fixture suite, фиксируем финальные метрики в CHANGELOG.md
- README.md по разделам §6 TASK.md (architecture, store choice, extraction, recall, evolution, tradeoffs, failure modes, how to run tests)
- Mermaid диаграмма архитектуры
- Таблица "что я бы сделал на 3-й день": graph traversal для multi-hop, learned-to-rank, embedding fine-tune, opinion-arc tracking

**Метрика:** README прочитывается за 5 минут и реально объясняет систему.

**CHANGELOG: v1.0 — Submission. Final recall@5 = X.XX. Final p95 latency = Y ms.**

---

## 6. Self-eval fixture (`fixtures/`)

5 conversations (per §7 TASK.md), каждая в JSON со списком turns + probes:

| Файл | Что проверяет | Probes (примеры) |
|---|---|---|
| `conv_career.json` | Supersession (employment) | "Where does the user work now?" → Notion (not Stripe) |
| `conv_pets.json` | Implicit facts | "What's the user's dog's name?" → Biscuit |
| `conv_preferences.json` | Opinion evolution | "How does the user feel about TypeScript?" → expects nuanced |
| `conv_multihop.json` | Multi-hop | "Where does the dog owner live?" → joins pet+location |
| `conv_noise.json` | Noise resistance | "What's the user's favorite color?" → empty/no hallucination |

`fixtures/probes.yaml`:
```yaml
- conv: conv_career
  query: "Where does the user work?"
  must_contain: ["Notion"]
  must_not_contain: ["Stripe"]   # стара информация — не должна доминировать
  is_multi_hop: false
- conv: conv_multihop
  query: "What city does the user with the dog Biscuit live in?"
  must_contain: ["Berlin"]
  is_multi_hop: true
- conv: conv_noise
  query: "What's the user's favorite color?"
  must_contain: []
  must_be_empty: true
```

`tests/test_recall_quality.py` ингестит conv'ы → бьёт probes → считает:
- **recall@k**: доля probes где `must_contain` все попали в context
- **precision**: доля citations что реально релевантны (manually labeled in fixture)
- **noise_score**: для `must_be_empty` — context должен быть пустой строкой

---

## 7. Контракт-тесты (`tests/test_contract.py`)

Минимум, требуемый §7 TASK.md, **плюс** edge-cases:
- ✅ `GET /health` → 200
- ✅ Roundtrip: POST /turns → GET /memories видит факт → POST /recall возвращает его
- ✅ Restart persistence: write turns → `docker compose restart app` → recall видит их
- ✅ Concurrent sessions: 2 user_id одновременно — нет bleed
- ✅ Cross-session same user: shared memory работает (документируем в README что это intentional)
- ✅ Malformed JSON → 400
- ✅ Missing required field (session_id) → 422
- ✅ Oversized payload (10MB) → 413
- ✅ Unicode/emoji → 201, нормально хранятся и достаются
- ✅ Cold session: POST /recall на пустой DB → `{"context":"","citations":[]}` 200
- ✅ Auth: токен задан в env → запрос без header → 401; токен не задан → header игнорится
- ✅ DELETE /sessions/{id} → 204; /memories больше не видит facts из этой сессии
- ✅ DELETE /users/{id} → 204; cascade в turns/messages/memories

---

## 8. CHANGELOG strategy

**Формат каждой записи** (per §6 TASK.md):
```markdown
## v0.X — <название итерации>

**What changed:** конкретные изменения в коде/архитектуре

**Why:** какой именно фикстурный pain points мотивировал

**Result:**
  - recall@5: X.XX → Y.YY
  - p95 latency: A ms → B ms
  - <конкретные probes которые стали проходить / ломаться>

**Next:** что осталось болеть — в очередь
```

Минимум **5–7 значимых entries** к моменту submission. Цель — показать инжиниринговый процесс, не финальное число.

---

## 9. Failure modes (для README раздел §6.7)

| Сценарий | Поведение |
|---|---|
| Alem LLM down / 5xx | Retry 3x exp backoff; затем сохранить turn БЕЗ extraction (raw сообщения остаются recallable через `messages.content_tsv` fallback) |
| Embeddings down | `/turns` возвращает 503 (без эмбеддинга нельзя класть в vector index) |
| Reranker down | Skip rerank stage, отдаём top-10 из RRF (graceful degradation) |
| API ключи отсутствуют | `/health` возвращает 200 но с `"degraded":["llm","embed"]`; /turns в degraded режиме сохраняет raw, не extract'ит |
| Postgres недоступен | `/health` 503; все endpoint'ы 503 |
| Слишком большой turn | 413 Payload Too Large |
| Юникод в prompt'е к LLM | Парсим JSON безопасно; на parse error → пустой `memories[]`, факты не теряются (turn всё равно сохранён) |

---

## 10. Что НЕ делаем (out of scope per §12 TASK.md)

- ❌ Multi-tenant prod-readiness
- ❌ Horizontal scale proofs
- ❌ Migration story (один `init.sql`)
- ❌ UI / agent-side code
- ❌ Async orchestration внутри /turns (60s timeout есть — синхронно ОК)
- ❌ Knowledge graph traversal (упомянем в README "what I'd do on day 3")

---

## 11. Open questions (по необходимости — спрошу до старта)

Сейчас открытых вопросов нет — все ключи получены, стек зафиксирован. Стартую с Step 0.

Если в процессе всплывёт что-то материальное — сразу останавливаюсь и пишу.

---

## 12. Estimated total time

| Шаг | Время |
|---|---|
| 0. Scaffold | 0.5ч |
| 1. Persistence + /turns raw | 1ч |
| 2. Naive recall baseline | 1ч |
| 3. LLM extraction | 2ч |
| 4. Hybrid retrieval | 1.5ч |
| 5. Reranker | 1ч |
| 6. Supersession | 2ч |
| 7. Multi-hop | 1.5ч |
| 8. Budget assembly | 1.5ч |
| 9. Robustness | 1ч |
| 10. README + finalize | 1ч |
| **Итого** | **~14 часов** |

С запасом укладываемся в 2 дня по 8 часов.

---

**Готов стартовать со Step 0 по твоей команде.**
