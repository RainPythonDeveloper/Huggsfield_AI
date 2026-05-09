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
