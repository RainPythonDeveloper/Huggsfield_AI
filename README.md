# memory-service

A Dockerized memory service for an AI agent — built for the Higgsfield engineering challenge.

> ⚠️ **Work in progress.** This README is a stub and will be replaced with the full architecture writeup at the end of Step 10. See [PLAN.md](PLAN.md) for the iterative roadmap and [CHANGELOG.md](CHANGELOG.md) for design history.

## Quick start

```bash
cp .env.example .env
# fill in ALEM_API_KEY, EMBED_API_KEY, RERANK_API_KEY
docker compose up -d
until curl -sf http://localhost:8080/health; do sleep 1; done
```

## Stack
- Python 3.12 + FastAPI
- Postgres 16 + pgvector + tsvector (single container, named volume)
- Alem AI for LLM (`alemllm`), embeddings (`text-1024`, dim=1024), rerank (`reranker`)

## Endpoints
Per TASK.md §3 contract — implemented incrementally across Steps 1–9. Full reference will live here at submission.
