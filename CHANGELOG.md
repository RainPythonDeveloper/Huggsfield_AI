# CHANGELOG

> Iteration history for the memory-service. Each entry follows the format:
> **What changed → Why → Result (with metrics) → Next.**
> Per TASK.md §6, the CHANGELOG is the most important deliverable for the human review.

---

## v0.1 — Boots, schema in place, no logic yet (2026-05-09)

**What changed:**
- Repo scaffolded: `Dockerfile`, `docker-compose.yml`, `pyproject.toml`, `.env.example`.
- Two-container compose: `db` (`pgvector/pgvector:pg16`) + `app` (FastAPI on port 8080).
- Postgres schema (`migrations/001_init.sql`) with three tables: `turns`, `messages`, `memories`.
  - `vector(1024)` column + HNSW index for ANN search.
  - `tsvector` GENERATED columns for BM25-style FTS.
  - `memories.supersedes` self-FK ready for fact-evolution chains.
- FastAPI lifespan manages asyncpg pool; `GET /health` pings DB and reports degraded LLM/embed/rerank flags if API keys missing.
- `pgdata` named volume → data survives `docker compose down`.

**Why:**
- Get the deploy story working *first* so each subsequent iteration is verified end-to-end via the eval harness's exact entrypoint (`docker compose up -d` + curl `/health`).
- Schema with both vector + tsvector columns chosen upfront so we don't need a migration when we add hybrid retrieval at Step 4.

**Result:**
- `docker compose up -d` boots cleanly. `curl localhost:8080/health` → `{"status":"ok","version":"0.1.0","degraded":["llm","embed","rerank"]}` until env keys provided.
- Restart-survival of DB schema confirmed: `docker compose down && up -d` keeps tables.
- No business logic yet; recall/extraction stubs come next.

**Next:**
- Step 1: implement `POST /turns` raw-store path + `DELETE /sessions/{id}` + `DELETE /users/{id}`. No extraction yet — just persistence.

---

## v0.2 — Raw turn storage + DELETE endpoints + global error handlers (2026-05-09)

**What changed:**
- Pydantic schemas for the full §3 contract in `schemas.py` (TurnIn/Out, RecallIn/Out, SearchIn/Out, MemoryOut, etc.).
- `repository.py` — async CRUD over asyncpg: `insert_turn` (atomic turn + messages in one transaction), `delete_session`, `delete_user`, `list_user_memories`.
- Routes: `POST /turns`, `DELETE /sessions/{id}`, `DELETE /users/{id}`, `GET /users/{user_id}/memories`. Stubs for `/recall` and `/search` (return empty — wired in Steps 2-5).
- Optional bearer auth dependency: gated by `MEMORY_AUTH_TOKEN` env. If unset, `Authorization` header ignored.
- Global FastAPI handlers: `RequestValidationError → 422` with structured detail; unhandled `Exception → 500` (no traceback leak). Per TASK.md §5 "service must not crash on malformed input".

**Why:**
- Persistence-first: data must be in Postgres before any business logic runs against it. Eval harness does `POST /turns` → `GET /memories`/`POST /recall` and expects synchronous correctness — we land that contract first, then layer extraction/recall.
- Stub `/recall` + `/search` so the eval harness gets 200s with valid JSON shape from day one (no crashes on cold sessions per §3).
- Auth-as-dependency keeps the gate centralized — flipping `MEMORY_AUTH_TOKEN` toggles all routes at once.

**Result:**
- Smoke test: `POST /turns` → 201 + UUID; row visible in `turns` + `messages` tables.
- `DELETE /sessions/{id}` → 204; only target session removed (other session for same user untouched).
- `DELETE /users/{id}` → 204; cascade removes turns + messages; `count(*) = 0` after.
- Malformed JSON body → 422 with `{"error":"validation_error","detail":[...]}` instead of stacktrace.
- Missing required fields (no `session_id`, empty `messages`, no `timestamp`) → 422 with all three reported.
- `/users/user-1/memories` → `{"memories":[]}` (correct: extraction not yet wired).

**Next:**
- Step 2: turn raw messages into embedded vectors (no LLM extraction yet) and stand up naive embedding-only `/recall` and `/search` to establish a recall@5 baseline against the fixture.

---

## v0.3 — Naive embedding recall, fixture-based eval, baseline established (2026-05-09)

**What changed:**
- `clients/embeddings.py`: Alem `text-1024` (dim=1024) wrapper with tenacity retry + http2.
- Migration `002_messages_embedding.sql`: adds `messages.embedding vector(1024)` + HNSW idx.
- `migrate.py`: idempotently re-applies migrations on every container boot (Postgres `docker-entrypoint-initdb.d` only runs on a fresh data dir, so this complements it).
- `services/ingest.py`: `POST /turns` now embeds every message and stores the vector synchronously before returning 201. Per TASK.md §5 *"after /turns returns, ingested data must be immediately available"*.
- `services/recall.py`: `/recall` and `/search` do `embed(query) → cosine top-k` against `messages`.
- 5 fixture conversations (`conv_career`, `conv_pets`, `conv_preferences`, `conv_multihop`, `conv_noise`) + 12 probes in `probes.yaml` with `must_contain` / `must_be_empty` / `is_multi_hop` flags.
- `tests/test_recall_quality.py`: ingests every fixture, runs every probe, prints recall@5 with category breakdowns.
- `tests/test_contract.py`: 7 contract tests covering roundtrip / cold session / malformed JSON / missing fields / unicode / concurrent-session isolation.

**Why:**
- Embedding the **raw** message (not yet structured memory) lets us measure how far a vanilla setup goes before LLM extraction is added — that's the comparison point for Step 3.
- Naming the v0.3 *baseline* is critical: every later step is now a delta against numbers we wrote down, not vibes.
- HNSW index on a 1024-dim column needs to exist from day 1 — adding it later forces a reindex over the whole corpus.

**Result:**
- **recall@5 = 9/12 = 75%** overall on the fixture.
  - Multi-hop: 2/2 = 100% — surprising! With a small fixture, naive cosine pulls in both relevant turns. This will degrade with larger corpora; treating it as “solved” at this stage would be a mistake.
  - Noise resistance: 0/2 = 0% — vanilla cosine top-k *always* returns its k best, even when none are relevant. **This is the cleanest gap in v0.3 to fix.**
- 7/7 contract tests green: roundtrip, cold session, malformed JSON, missing fields, unicode, concurrent-session isolation.
- Failed probes:
  - `career_role`: needs canonical role normalization ("product management" ≠ "product manager") — Step 3 LLM extraction territory.
  - `noise_color`, `noise_food`: see noise resistance above.
- p95 ingest latency: ~150ms / message embedded (sequential httpx calls; can parallelize in Step 3 batch).
- p95 recall latency: ~110ms (single embed + 1 SQL query).

**Next:**
- Step 3: LLM extraction. Replace "embed every raw message" with "extract structured facts → embed those". This should both improve precision (canonical `key=role,value=Product Manager` beats free-text matching) and start populating `/users/{id}/memories` with structured records, which is the single biggest visible quality differentiator on the human review.


