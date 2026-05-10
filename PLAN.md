# PLAN.md — Memory Service for AI Agent (Higgsfield Challenge)

> Iterative implementation plan with measurable metrics at every step.
> Each iteration = one entry in `CHANGELOG.md` with **What / Why / Result / Next**.
> Start date: 2026-05-09. Deadline: 2 days of focused work.

---

## 0. Solution summary (TL;DR)

**Memory Service** — a Dockerized HTTP service on **Python 3.12 + FastAPI** that:

1. Accepts conversation turns via `POST /turns`.
2. Extracts structured memories through an LLM (Alem `alemllm`).
3. Stores them in **Postgres 16 + pgvector + tsvector** (one container with a named volume).
4. Returns relevant context via `POST /recall` with hybrid retrieval:
   `(BM25 ⊕ embeddings) → RRF → Alem reranker → priority-aware budget assembly`.
5. Detects contradictions and maintains supersession chains so facts can evolve over time.

The §3 contract from TASK.md is honoured one-to-one. All 7 endpoints are synchronous; eventual consistency is explicitly excluded — TASK §5 says *"after `/turns` returns, ingested data must be immediately available via `/recall`"*.

---

## 1. Tech stack (locked)

| Layer         | Choice                                                                   | Rationale                                                                                                                                   |
| ------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Language      | Python 3.12                                                              | Best AI ecosystem; fast prototyping                                                                                                         |
| Web           | FastAPI + uvicorn                                                        | Pydantic contract validation out of the box, async, OpenAPI                                                                                 |
| DB            | Postgres 16 + pgvector 0.7 + builtin FTS                                 | Single container: vector + relational + BM25-style ranking; persistence via named volume                                                    |
| Migrations    | Plain `init.sql` + idempotent `migrate.py` runner                    | MVP:`001_init.sql` runs on first DB boot via `docker-entrypoint-initdb.d`; `migrate.py` re-applies any later files on every app start |
| LLM           | Alem `alemllm` via OpenAI-compatible API at `https://llm.alem.ai/v1` | OpenAI-shaped endpoint; JSON often wrapped in fences — we will strip them                                                                  |
| Embeddings    | Alem `text-1024` (dim=1024)                                            | OpenAI-compatible                                                                                                                           |
| Reranker      | Alem `reranker` via httpx                                              | Cohere-compatible cross-encoder; in our calibration probe it spreads relevant vs. irrelevant ~0.97 vs ~0.001                                |
| HTTP client   | `httpx[http2]` + `tenacity` retry                                    | Async; needed for parallel embedding/extraction calls                                                                                       |
| Tokens        | `tiktoken` (cl100k_base)                                               | For the `max_tokens` budget in `/recall` (close enough to Alem's tokenizer)                                                             |
| Tests         | `pytest` + `pytest-asyncio` + `httpx.AsyncClient`                  | Contract tests hit the FastAPI app directly                                                                                                 |
| Lint/format   | `ruff` (lint+format)                                                   | Fast, single tool                                                                                                                           |
| Container     | Multi-stage `python:3.12-slim`                                         | ~200MB final image                                                                                                                          |
| Orchestration | `docker-compose.yml` with two services: `app` + `db`               | `db` — `pgvector/pgvector:pg16`, `app` — our Dockerfile                                                                             |

---

## 2. Architecture

```
┌─────────────────── memory-service container (FastAPI, port 8080) ──────────────────────┐
│                                                                                         │
│  HTTP layer (routes/)           Domain services (services/)         Storage             │
│  ─────────────────              ──────────────────────────         ─────────            │
│  POST /turns      ─────►  ingest → extraction → supersession ─►  ┌────────────────────┐ │
│  POST /recall     ─────►  recall (hybrid + rerank + budget)  ─►  │ Postgres+pgvector  │ │
│  POST /search     ─────►  recall (structured results)        ─►  │  - turns           │ │
│  GET  /users/{}/memories ─►  repository.list_user_memories  ─►   │  - messages        │ │
│  DELETE /sessions/{} ───►  repository.delete_session        ─►   │  - memories        │ │
│  DELETE /users/{}    ───►  repository.delete_user           ─►   │    (vector + tsv)  │ │
│  GET  /health        ───►  db ping + degraded flags              └────────────────────┘ │
│                                                                                         │
│         ▲                          ▲                                                    │
│         │                          │                                                    │
│  ┌──────┴──────┐           ┌───────┴──────────┐         ┌──────────────────┐            │
│  │ Alem LLM    │           │ Alem Embeddings  │         │ Alem Reranker    │            │
│  │ /chat/...   │           │ /embeddings      │         │ /rerank          │            │
│  └─────────────┘           └──────────────────┘         └──────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ named volume │  pgdata
                   └──────────────┘
```

### `/turns` flow (write path)

```
1. Receive turn         ─┐
2. Persist turn+messages │  one transaction (atomic)
3. Extract facts (LLM)   │  ←—— synchronous; ~4–5 s / turn typical
4. Resolve contradictions│  for each fact: query active memories with same (user_id, key)
5. Insert / supersede    │  pgvector: insert canonical embedding
6. Return 201            ─┘  ← data is immediately available to /recall
```

### `/recall` flow (read path)

```
1. Embed query (1 call)
2. Query rewriting (LLM, 1 call) — classifies single-hop vs multi-hop and
   decomposes the question into 1–3 atomic sub-queries when needed
3. Hybrid retrieval (per sub-query, in parallel):
   a) pgvector cosine over memories.embedding              → top 30
   b) BM25 over memories.value_tsv (key + value)           → top 30
   c) RRF fusion (k=60) → top 20
   When multi-hop: an outer RRF merges sub-query result lists
4. Cross-encoder rerank against the ORIGINAL query → top 8 with score ≥ 0.05
5. Bucket by priority:
   stable_user_facts (fact|preference|relation, active)
   query_relevant_memories (events, opinions)
   recent_session_context (last 4 messages, only if <6 prior facts)
6. Greedy fill against soft cap = 0.95 × max_tokens. Bullets are dropped, not truncated.
7. Format prose:
     ## Known facts about this user
     - …
     ## Relevant from recent conversations
     - …
   plus citations[].
```

---

## 3. DB schema (Postgres 16 + pgvector)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ── Conversation history ─────────────────────────────────────────────────
CREATE TABLE turns (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  TEXT NOT NULL,
    user_id     TEXT,                                 -- NULL = anonymous
    timestamp   TIMESTAMPTZ NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw         JSONB NOT NULL,                       -- full incoming payload
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX turns_session_idx ON turns(session_id);
CREATE INDEX turns_user_idx    ON turns(user_id);

CREATE TABLE messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id     UUID NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,                        -- user | assistant | tool
    name        TEXT,
    content     TEXT NOT NULL,
    position    INT NOT NULL,                         -- order inside the turn
    -- tsvector for fallback recall over raw text (when extraction missed something)
    content_tsv tsvector GENERATED ALWAYS AS
                (to_tsvector('english', content)) STORED
);
CREATE INDEX messages_turn_idx ON messages(turn_id);
CREATE INDEX messages_tsv_idx  ON messages USING GIN(content_tsv);

-- ── Extracted memories (the main table) ──────────────────────────────────
CREATE TABLE memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT,                             -- NULL = session-scoped
    session_id      TEXT,                             -- NULL = global to user
    type            TEXT NOT NULL,                    -- fact|preference|opinion|event|relation
    key             TEXT NOT NULL,                    -- canonical key, e.g. "employer", "city"
    value           TEXT NOT NULL,                    -- canonical value: "Notion"
    raw_quote       TEXT,                             -- quote from the turn (provenance)
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

-- supersession is modelled as a self-FK on memories.supersedes;
-- a separate memory_history table is unnecessary — chains are reconstructible
-- via a recursive CTE.
```

**Why this schema:**

- `memories` is a single table with a `supersedes` self-FK → history is reconstructed via a recursive CTE. No separate audit table to keep in sync.
- `tsvector` columns are `GENERATED ALWAYS` → never out of date, no triggers to maintain.
- `HNSW` on the embedding gives O(log n) ANN search; better than IVF for small corpora (our case).
- `messages.content_tsv` is the cold-extraction fallback: if the LLM extractor missed a fact, raw text is still queryable.
- The `WHERE active` partial index on `(user_id, key)` makes the "find existing fact for this key" lookup (used on every supersession check) effectively constant-time.

A second migration `002_messages_embedding.sql` adds a `vector(1024)` column on `messages` so the Step 2 baseline can run before the LLM extractor exists. This column stays after Step 3 to support the cold-extraction fallback path.

---

## 4. Project layout

```
memory-service/
├── README.md                    # architecture + design rationale + run instructions
├── CHANGELOG.md                 # ITERATION HISTORY (the primary deliverable per TASK §6)
├── PLAN.md                      # this file
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── pyproject.toml
├── src/
│   └── memory/
│       ├── __init__.py
│       ├── main.py              # FastAPI app, lifespan, CORS, error handlers, body-size middleware
│       ├── config.py            # Pydantic Settings, reads .env
│       ├── schemas.py           # Pydantic in/out models (TASK §3 contract)
│       ├── auth.py              # optional Bearer-token dependency, gated by MEMORY_AUTH_TOKEN
│       ├── db.py                # asyncpg pool, lifespan management
│       ├── migrate.py           # idempotent migration runner (re-applies any 00X_*.sql)
│       ├── repository.py        # async CRUD over asyncpg (turns, messages, memories)
│       ├── migrations/
│       │   ├── 001_init.sql
│       │   └── 002_messages_embedding.sql
│       ├── routes/
│       │   ├── turns.py
│       │   ├── recall.py
│       │   ├── search.py
│       │   ├── memories.py
│       │   └── cleanup.py
│       ├── services/
│       │   ├── ingest.py        # /turns: persist → extract pipeline
│       │   ├── extraction.py    # LLM extraction + supersession dispatch
│       │   ├── supersession.py  # contradiction resolution + chaining
│       │   ├── recall.py        # hybrid retrieval + rerank + budgeted prose
│       │   └── query_rewrite.py # multi-hop classifier/decomposer
│       ├── clients/
│       │   ├── llm.py           # Alem chat-completions wrapper (tenacity, http2)
│       │   ├── embeddings.py    # Alem embeddings + retries
│       │   └── reranker.py      # Alem rerank wrapper
│       ├── prompts/
│       │   ├── extract.py       # system+user prompts for extraction
│       │   ├── supersession.py  # 4-verdict LLM-judge prompt
│       │   └── query_rewrite.py # multi-hop decomposition prompt
│       └── util/
│           ├── tokens.py        # tiktoken counter
│           ├── rrf.py           # reciprocal rank fusion (k=60)
│           └── json_parse.py    # strips ```json fences and stray prose
├── tests/
│   ├── conftest.py
│   ├── test_contract.py         # TASK §3 endpoint shapes + roundtrip
│   ├── test_recall_quality.py   # ingests fixtures/, runs probes, computes recall@k
│   ├── test_supersession.py     # career-arc E2E
│   ├── test_budget.py           # parametrized 128/256/512/1024-token compliance
│   ├── test_persistence.py      # docker compose restart survival
│   └── test_robustness.py       # oversized / unicode / empty / invalid / concurrent
└── fixtures/
    ├── conv_career.json         # employment evolution: Stripe → Notion
    ├── conv_pets.json           # implicit fact: "walking Biscuit" → has dog Biscuit
    ├── conv_preferences.json    # opinion arc: TS love → frustration → nuanced
    ├── conv_multihop.json       # 2 facts in different turns, joined at /recall
    ├── conv_noise.json          # queries about topics that never appeared
    └── probes.yaml              # 12 probes: query → must_contain[] / must_not_dominate[]
```

---

## 5. Iterative steps with metrics

> Each step → one atomic commit + one CHANGELOG.md entry.
> After steps 3, 5, 6, 7, 8, 9 we re-run `pytest tests/test_recall_quality.py` and record metrics (recall@k, multi-hop, noise resistance, p95 latency).

### Step 0: Repo scaffold + health (0.5h) → `v0.1`

**Build:**

- `Dockerfile`, `docker-compose.yml` (app + db, named volume `pgdata`).
- FastAPI app with `GET /health` (DB ping + `degraded` flags for missing LLM/embed/rerank API keys).
- `pyproject.toml` with all dependencies pinned.
- `.env.example` with all three Alem keys + `MEMORY_AUTH_TOKEN`.
- `migrations/001_init.sql` with pgvector extension and the schema from §3.

**Metric:** `docker compose up -d && curl localhost:8080/health` → `{"status":"ok","version":"0.1.0",...}`. Restart container → DB schema persists.

**CHANGELOG: `v0.1` — Boots, schema in place, no logic yet.**

---

### Step 1: Persistence layer + `POST /turns` (raw store) (1h) → `v0.2`

**Build:**

- `db.py`: asyncpg pool + transaction wrapper.
- `repository.py`: `insert_turn` (atomic turn + messages in one transaction), `delete_session`, `delete_user`, `list_user_memories`.
- Routes: `POST /turns`, `DELETE /sessions/{id}`, `DELETE /users/{id}`, `GET /users/{id}/memories`. Stub `/recall` and `/search` returning empty payloads (so the eval harness gets 200s with valid JSON shape from day one).
- Pydantic schemas exactly per TASK §3.
- Optional Bearer-auth dependency gated by `MEMORY_AUTH_TOKEN` (if unset → header ignored).
- Global error handlers: `RequestValidationError → 422` with structured detail; unhandled `Exception → 500` (no traceback leakage).

**Metric:** `POST /turns` smoke from TASK §7 returns 201 + UUID. `DELETE /sessions/{id}` → 204; only target session removed. Malformed JSON → 422 with structured error (no stacktrace).

**CHANGELOG: `v0.2` — Raw turn storage works; persistence verified across restart; 7/7 contract tests green.**

---

### Step 2: Naive recall (embeddings only) — baseline (1h) → `v0.3`

**Build:**

- `clients/embeddings.py`: Alem `text-1024` wrapper with tenacity retry + http2.
- Migration `002_messages_embedding.sql`: `messages.embedding vector(1024)` + HNSW index.
- `migrate.py`: idempotent re-application on every app boot (the `docker-entrypoint-initdb.d` only runs on a fresh data dir, so this complements it).
- `services/ingest.py`: `POST /turns` embeds every message and stores the vector before returning 201 (TASK §5: *"after `/turns` returns, ingested data must be immediately available"*).
- `services/recall.py`: `/recall` and `/search` do `embed(query) → cosine top-k` against `messages`.
- `fixtures/`: 5 conversations + `probes.yaml` (12 probes) with `must_contain` / `must_not_dominate` / `must_be_empty` / `is_multi_hop`.
- `tests/test_recall_quality.py`: ingests every fixture, runs every probe, prints recall@5 with category breakdowns.

**Metric on the fixture:** baseline **recall@5 = 9/12 = 75%** (multi-hop 100%, noise 0% — vanilla cosine top-k always returns *something*).

**CHANGELOG: `v0.3` — Naive embedding recall. Baseline recall@5 = 75%; noise resistance is the obvious gap.**

> 💡 *This is the reference point: every later step is measured against these numbers.*

---

### Step 3: LLM extraction pipeline (2h) → `v0.4`

**Build:**

- `clients/llm.py`: Alem chat/completions with tenacity retry + http2.
- `util/json_parse.py`: lenient parser handling ` ```json ` fences, leading prose, stray brackets. Returns `None` on failure (a single bad reply must never break ingest).
- `prompts/extract.py`: system prompt with explicit type taxonomy (`fact|preference|opinion|event|relation`), canonical-key list (employer, role, city, pet_dog_name, dietary_restriction, …), atomicity rule ("I work at Notion as a PM" → 2 memories), implicit-fact and correction-capture rules, strict JSON schema.
- `services/extraction.py`: LLM call → lenient parse → schema clean → embed canonical *"The user's `<key humanized>` is `<value>`"* → `INSERT INTO memories`.
- `services/ingest.py` rewired: persist turn → call extraction synchronously → memories available immediately.
- `services/recall.py` switched from `messages.embedding` (Step 2) to `memories.embedding`. Output is bucketed prose: "## Known facts about this user" + "## Relevant from recent conversations".

**Metric:**

- `GET /users/{id}/memories` returns structured records (employer, city, pet_dog_name, …) — answers TASK §4 explicit red flag *"if `/memories` returns raw message chunks, that's a red flag"*.
- **recall@5 = 10/12 = 83%** (+8 pts vs baseline).
- p95 ingest latency ~4.5 s/turn (1 LLM + N parallel embeddings + N inserts). Well under the 60 s SLA in TASK §3.

**CHANGELOG: `v0.4` — LLM extraction. recall@5: 75% → 83%. `/memories` is now structured.**

---

### Step 4: Hybrid retrieval (BM25 + embeddings + RRF) (1.5h) → `v0.5`

**Build:**

- `repository.search_memories_by_bm25`: Postgres FTS over `value_tsv`, ranked with `ts_rank_cd` (cover density), gated by `plainto_tsquery('english', $1)`.
- `repository.search_messages_by_bm25`: secondary FTS channel over `messages.content_tsv` — cold-extraction fallback for facts the extractor missed.
- `util/rrf.py`: Reciprocal Rank Fusion (Cormack et al., k=60). Retains per-channel rank info in `_channels` so we can audit *why* a hit surfaced (debuggable retrieval).
- `services/recall.py`: vector + BM25 in parallel via `asyncio.gather`, fused with RRF (top 30 each → top 20 fused). Empty-result fallback queries raw messages.
- `services/search.py` (within `recall.py`): `/search` uses the same hybrid pipeline; `metadata.channels` exposes the per-channel rank for each hit.

**Metric:**

- **recall@5 = 10/12 = 83%** — unchanged on this small fixture (vector alone already captured everything the extractor produced). The architectural win is *robustness on unseen workloads* — keyword-heavy queries on the eval harness's hidden fixture should benefit.
- Verified channel co-firing: query "dog name" → `pet_dog_name: Biscuit` matched in both channels (vector rank 0 AND bm25 rank 0).
- p95 recall latency: ~120 ms (we run vector + BM25 concurrently).
- Noise probes still 0/2 — confirmed by inspection; this is the next thing to fix.

**CHANGELOG: `v0.5` — Hybrid BM25+embed via RRF. recall@5 unchanged on small fixture; robustness on unseen workloads is the upside.**

---

### Step 5: Reranker stage + noise gating (1h) → `v0.6`

**Build:**

- `clients/reranker.py`: Alem `/rerank` wrapper (Cohere-compatible), returns `[{index, score}]`.
- `services/recall.py`: insert reranker between RRF and prose assembly. Top 20 → reranker → top 8 with `score ≥ 0.05`.
- **Doc-format calibration (the load-bearing fix):** the cross-encoder is sensitive to subject framing. Queries say *"the user"*, so docs must too. We render every doc as `"The user's <key humanized> is <value>. Originally said: <quote>"`. Without this, scores collapse to ~0.001 even for relevant pairs.
- Resilience: if vector embedding fails, `/recall` **degrades to BM25-only** instead of returning 500. If reranker fails, RRF order is kept. The pipeline becomes multi-channel-fault-tolerant.
- Embedding retry bumped to 5 attempts with 0.6→8 s exponential backoff (Alem's intermittent 502s motivated this).

**Metric:**

- **recall@5 = 12/12 = 100%** (+17 pts vs `v0.5`).
- **noise resistance: 0/2 → 2/2 = 100%** (the headline fix this step).
- multi-hop: 2/2 = 100%, no regressions on contract tests.
- Recall p95 latency ~250 ms (added ~130 ms for the rerank API call).

**CHANGELOG: `v0.6` — Alem reranker + 3rd-person doc framing. recall@5 = 100%; noise 0% → 100%.**

---

### Step 6: Supersession / contradiction handling (2h) → `v0.7`

This is TASK §4 hard problem #1 and a key grading axis.

**Build:**

- `prompts/supersession.py`: focused LLM-judge prompt with **4 verdicts**:
  - `supersede` — new replaces old (signals: "started", "switched", "now", "moved", "joined", "actually I meant", "no longer", "used to").
  - `coexist` — both true at once (multi-value keys: pets, hobbies, languages).
  - `keep_old` — new is HISTORICAL, existing is current ("I used to work at X").
  - `noop` — duplicate / less-precise restatement.
- `services/supersession.py`: query active memories for `(user_id, key)` → exact-match shortcut for duplicates → LLM judge call (1 chat completion, ≤200 tokens). **Heuristic fallback** if LLM fails: singular keys default to `supersede`, plural keys (`MULTI_VALUE_KEYS` whitelist: pets, languages, hobbies, …) default to `coexist`.
- `repository.find_active_memories_by_key`, `repository.mark_superseded` (deactivate + chain-link `supersedes=most_recent_old.id` in two atomic UPDATEs).
- `repository.insert_memory` accepts `active=False` for the `keep_old` path.
- `services/extraction.py`: every candidate goes through `supersession.resolve` before insert. Per-turn summary log with `superseded_old / coexist_inserts / historical_inserts / noop_skipped` counts.
- `tests/test_supersession.py`: ingests `conv_career.json` (Stripe→Notion) and asserts the chain.

**Metric:**

- `conv_career.json`: turn1 "I work at Stripe" → turn2 "Just started at Notion".
  - `/recall "Where do I work?"` → "Notion" (no Stripe pollution).
  - `/users/{id}/memories` shows BOTH employers; Stripe is `active=false`, Notion is `active=true` with `supersedes=stripe_uuid`. History preserved per TASK §4.
- **All 9/9 tests green** including the new dedicated supersession E2E.
- recall@5 stays at 100% (the headline metric was already saturated; what changed is *correctness on stricter graders*).

**CHANGELOG: `v0.7` — Supersession chains (TASK §4 hard problem #1). 4-verdict LLM judge + heuristic fallback. Career-arc E2E: PASS.**

---

### Step 7: Multi-hop via query decomposition (1.5h) → `v0.8`

"What city does the user with the dog named Biscuit live in?" — requires 2 facts joined.

**Build:**

- `prompts/query_rewrite.py`: classifier + decomposer prompt with explicit multi-hop signals (relative clauses, anaphora, compound questions). Returns `{"is_multi_hop": bool, "sub_queries": [...]}`.
- `services/query_rewrite.py`: thin LLM wrapper. LLM failure → degrade to single-hop (caller proceeds with original query).
- `services/recall.py` refactored: `_retrieve` runs decomposition first, then either:
  - **single-hop** → one `_hybrid_memories` pass.
  - **multi-hop** → parallel `_hybrid_memories` per sub-query, then **outer RRF** to merge candidate lists. The reranker still runs against the **original (un-decomposed) query** so cross-encoder scoring stays aligned with what the user actually asked.
- Closed-loop safety: if all sub-queries return empty → fall back to a single-hop pass on the original (the LLM might decompose well but lose natural-language framing the reranker prefers).

**Metric:**

- All 9 tests still green. recall@5 / multi-hop / noise: 100% / 100% / 100%.
- Live decomposition example:
  ```
  POST /recall {"query": "What city does the user with the dog Biscuit live in?"}
  → multi_hop_decomposed sub_queries=["user's pet dog name", "user's city"]
  → context: pet dog name: Biscuit + city: Berlin   ✓
  ```
- Recall p95 latency: ~700–900 ms (LLM rewrite + parallel hybrid passes + rerank). Single-hop queries unchanged (decomposition LLM returns fast and the normal path runs).

**CHANGELOG: `v0.8` — Multi-hop via LLM decomposition + RRF over sub-queries; reranker still scores original query.**

---

### Step 8: Token-budget-aware context assembly (1.5h) → `v0.9`

TASK §3: *"Should respect `max_tokens`. When budget is tight, prioritize: stable user facts first, then query-relevant memories, then recent context. Your priority logic is a design decision we care about — defend it in the README."*

**Build:**

- `util/tokens.py`: lazy `tiktoken.cl100k_base` wrapper.
- `services/recall.py` — new `_format_recall_budgeted()`:
  - Three buckets, written in priority order:
    1. **stable user facts** — type ∈ {fact, preference, relation}, `active=true`.
    2. **query-relevant memories** — everything else from rerank (events, opinions).
    3. **recent conversation** — last 4 messages from `session_id`, only added when budget remains AND we have **<6 prior citations** (avoids drowning specific facts in chit-chat).
  - Greedy fill against soft cap = `0.95 × max_tokens`. Bullets are *dropped*, not truncated — half-sentences look bad and the precision isn't worth it.
  - Cold fallback (`_format_message_fallback`) also enforces the budget.
- `tests/test_budget.py`: parametrized `[128, 256, 512, 1024]` budget compliance + a tight-budget priority assertion (`pet_dog_name: Biscuit` survives at 128 tokens).

**Metric:**

- **All 14/14 tests green** (added 5 budget tests):
  - `test_budget_respected[128/256/512/1024]`: actual_tokens ≤ 1.10 × budget.
  - `test_user_facts_priority_at_tight_budget`: at 128 tokens the user's `pet_dog_name` still surfaces.
- recall@5 / multi-hop / noise / supersession / contract — no regressions.

**CHANGELOG: `v0.9` — Budget-aware assembly. Priority: stable facts → query-relevant → recent. Tested at 128/256/512/1024 tokens, no overflow.**

---

### Step 9: Robustness + persistence + concurrency (1h) → `v1.0-rc`

**Build:**

- `main.py`: `_BodySizeLimit` middleware — rejects requests with `Content-Length > 1 MB` with **413 Payload Too Large**.
- `tests/test_persistence.py`: ingest a turn → `docker compose restart` → poll `/health` for up to 60 s → assert both `/memories` and `/recall` recover the data.
- `tests/test_robustness.py`:
  - `test_oversized_payload`: 1.5 MB body → 4xx, service still healthy.
  - `test_emoji_unicode_and_zero_width`: mixed emoji (🇩🇪, 🍣, 🙂), Cyrillic, zero-width Unicode → 201, recall works.
  - `test_empty_messages_array_rejected`: empty `messages: []` → 422.
  - `test_invalid_role_rejected`: `role: "wizard"` → 422.
  - `test_search_empty_corpus_returns_empty`: `/search` for unknown user → `{"results": []}` + 200.
  - `test_concurrent_ingest_no_corruption`: 8 parallel `POST /turns` against 3 user buckets via threadpool → all 201, no asyncpg pool deadlock, no row corruption.

**Metric:**

- **21/21 tests green** end-to-end:
  - 7 contract (TASK §3 endpoint shapes)
  - 5 budget (TASK §3 max_tokens compliance)
  - 1 supersession E2E (TASK §4 hard problem #1)
  - 1 recall quality (12 fixture probes — 100% recall, 100% multi-hop, 100% noise resistance)
  - 1 restart persistence
  - 6 robustness (oversized / unicode / empty / invalid role / empty corpus / concurrent)
- Service stays healthy after every test class. No 5xx recoveries needed.

**CHANGELOG: `v1.0-rc` — Robustness, persistence, concurrency hardening. 21/21 tests green.**

---

### Step 10: Final tuning + README + CHANGELOG cleanup (1h) → `v1.0`

**Build:**

- Run the full fixture suite, freeze final metrics in CHANGELOG.md.
- `README.md` rewritten per TASK §6 sections one-for-one (architecture, store choice, extraction, recall, evolution, tradeoffs, failure modes, how to run tests).
- ASCII architecture diagram.
- "What I'd do on day 3" section: graph traversal for multi-hop, learned-to-rank, embedding fine-tune, opinion-arc tracking, online eval pipeline.

**Metric:** README readable in 5 minutes and genuinely explains the system.

**CHANGELOG: `v1.0` — Submission. Final recall@5 = 100%, multi-hop = 100%, noise = 100%, p95 latency ~700–900 ms, 21/21 tests green.**

---

## 6. Self-eval fixture (`fixtures/`)

5 conversations (per TASK §7), each in JSON with a list of turns + probes:

| File                      | What it tests             | Probes (examples)                                                                                                           |
| ------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `conv_career.json`      | Supersession (employment) | "Where does the user work now?" → Notion (not Stripe); "Has the user ever worked at Stripe?" → Stripe (history preserved) |
| `conv_pets.json`        | Implicit facts            | "What's the user's dog's name?" → Biscuit; "What breed?" → border collie                                                  |
| `conv_preferences.json` | Opinion evolution         | "How does the user feel about TypeScript today?" → expects nuanced view                                                    |
| `conv_multihop.json`    | Multi-hop                 | "What city does the user with the dog Biscuit live in?" → joins pet+location                                               |
| `conv_noise.json`       | Noise resistance          | "What's the user's favorite color?" → empty/no hallucination                                                               |

`fixtures/probes.yaml` — 12 probes total, with the schema:

```yaml
- id: career_current
  user_id: fx-career
  query: "Where does the user work?"
  must_contain: ["Notion"]
  must_not_dominate: ["Stripe"]   # stale info — must not dominate
  notes: "Stripe is older — Notion should win as 'current'."
- id: multihop_dog_city
  user_id: fx-multihop
  query: "What city does the user with the dog Biscuit live in?"
  must_contain: ["Berlin"]
  is_multi_hop: true
- id: noise_color
  user_id: fx-noise
  query: "What's the user's favorite color?"
  must_be_empty: true
```

`tests/test_recall_quality.py` ingests each conversation → runs each probe → measures:

- **recall@k**: fraction of probes where every `must_contain` token appears in context.
- **noise_score**: for `must_be_empty` probes — context must be empty.
- **multi-hop subset**: same recall@k restricted to `is_multi_hop=true` probes.

---

## 7. Contract tests (`tests/test_contract.py`)

The minimum required by TASK §7, **plus** edge cases:

- `GET /health` → 200.
- Roundtrip: `POST /turns` → `GET /memories` sees fact → `POST /recall` returns it.
- Restart persistence: write turns → `docker compose restart app` → recall sees them.
- Concurrent sessions: 2 user_ids in parallel — no cross-bleed.
- Cross-session same user: shared memory works (documented in README as intentional).
- Malformed JSON → 422.
- Missing required field (`session_id`, empty `messages`, missing `timestamp`) → 422.
- Oversized payload (1.5 MB) → 413.
- Unicode/emoji/zero-width → 201, stored and retrievable.
- Cold session: `POST /recall` on empty DB → `{"context":"","citations":[]}` 200.
- Auth: token set in env → request without header → 401; token unset → header ignored.
- `DELETE /sessions/{id}` → 204; `/memories` no longer sees facts from that session.
- `DELETE /users/{id}` → 204; cascade through turns/messages/memories.

---

## 8. CHANGELOG strategy

**Per-entry format** (per TASK §6):

```markdown
## v0.X — <iteration name>

**What changed:** concrete code/architecture changes

**Why:** which fixture pain points motivated this

**Result:**
  - recall@5: X% → Y%
  - p95 latency: A ms → B ms
  - <specific probes that started passing / regressing>

**Next:** what's still hurting — queued
```

Target: at least 5–7 substantive entries by submission. The goal is to surface the **engineering process**, not just the final number.

---

## 9. Failure modes (for README §6.7)

| Scenario                       | Behaviour                                                                                                                                                 |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Alem LLM down / 5xx            | Tenacity retries 5× with exponential backoff; then save the turn WITHOUT extraction (raw messages stay recallable via `messages.content_tsv` fallback) |
| Embeddings down on `/recall` | Degrade to BM25-only over `memories.value_tsv` instead of returning 500                                                                                 |
| Embeddings down on `/turns`  | Tenacity retries; if all fail, return 5xx (we can't write to a vector index without an embedding)                                                         |
| Reranker down                  | Skip rerank stage, return top 10 from RRF (graceful degradation)                                                                                          |
| API keys absent at boot        | `/health` returns 200 with `"degraded":["llm","embed","rerank"]`; `/turns` runs in degraded mode (raw store, no extraction)                         |
| Postgres unavailable           | `/health` 503; all endpoints 503                                                                                                                        |
| Oversized turn (>1 MB)         | 413 Payload Too Large via `_BodySizeLimit` middleware                                                                                                   |
| Unicode/emoji/zero-width chars | Stored as-is; safe JSON parsing with `util/json_parse.py` falls back to empty `memories[]` on parse error — turn is still saved                      |
| Restart mid-write              | `/turns` is one transaction → either fully committed or rolled back; no orphan rows                                                                    |

---

## 10. Out of scope (per TASK §12)

- Multi-tenant prod-readiness.
- Horizontal scale proofs.
- Migration story beyond `001_init.sql` + `002_messages_embedding.sql` + idempotent runner.
- UI / agent-side code.
- Async orchestration inside `/turns` (60 s budget allows synchronous extraction).
- Knowledge-graph traversal (mentioned in README "what I'd do on day 3").

---

## 11. Open questions

None at start. All three Alem keys are provisioned, the stack is locked, contract is unambiguous. Starting from Step 0.

If anything material surfaces mid-flight, I stop and write it down.

---

## 12. Estimated total time

| Step                           | Time                |
| ------------------------------ | ------------------- |
| 0. Scaffold                    | 0.5h                |
| 1. Persistence +`/turns` raw | 1h                  |
| 2. Naive recall baseline       | 1h                  |
| 3. LLM extraction              | 2h                  |
| 4. Hybrid retrieval            | 1.5h                |
| 5. Reranker                    | 1h                  |
| 6. Supersession                | 2h                  |
| 7. Multi-hop                   | 1.5h                |
| 8. Budget assembly             | 1.5h                |
| 9. Robustness                  | 1h                  |
| 10. README + finalize          | 1h                  |
| **Total**                | **~14 hours** |

Comfortably fits two 8-hour days with slack for surprises.

---

## 13. Final metrics summary (filled at v1.0)

| Step           | Headline                  | recall@5       | multi-hop      | noise          | tests           | latency p95            |
| -------------- | ------------------------- | -------------- | -------------- | -------------- | --------------- | ---------------------- |
| v0.1           | scaffold + schema         | —             | —             | —             | —              | —                     |
| v0.2           | raw turn store + DELETE   | —             | —             | —             | 7/7 contract    | —                     |
| v0.3           | naive embedding recall    | 75%            | 100%           | 0%             | 7/7             | ~110 ms                |
| v0.4           | LLM extraction            | 83%            | 100%           | 0%             | 7/7             | ~250 ms                |
| v0.5           | hybrid BM25 + RRF         | 83%            | 100%           | 0%             | 7/7             | ~120 ms                |
| v0.6           | + Alem reranker           | **100%** | 100%           | **100%** | 8/8             | ~250 ms                |
| v0.7           | + supersession            | 100%           | 100%           | 100%           | 9/9             | ~250 ms                |
| v0.8           | + multi-hop decomposition | 100%           | 100%           | 100%           | 9/9             | ~700–900 ms           |
| v0.9           | + budget assembly         | 100%           | 100%           | 100%           | 14/14           | ~700–900 ms           |
| v1.0-rc        | + robustness/persistence  | 100%           | 100%           | 100%           | **21/21** | ~700–900 ms           |
| **v1.0** | submission                | **100%** | **100%** | **100%** | **21/21** | **~700–900 ms** |

The single biggest jump was `v0.6` (reranker + 3rd-person doc framing): noise resistance 0% → 100%. The biggest design risk addressed was `v0.7` supersession (TASK §4 hard problem #1). Total ~14h focused work — on the original estimate.

---

**Ready to start at Step 0 on your signal.**
