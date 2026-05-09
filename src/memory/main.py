import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from memory import __version__
from memory.clients import embeddings as embed_client
from memory.clients import llm as llm_client
from memory.clients import reranker as rerank_client
from memory.config import get_settings
from memory.db import close_pool, init_pool, ping
from memory.migrate import apply_migrations
from memory.routes import cleanup, memories, recall, search, turns


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    log = logging.getLogger("memory.startup")
    log.info("starting", extra={"version": __version__})
    await init_pool()
    await apply_migrations()
    log.info("db_ready")
    yield
    await embed_client.close_clients()
    await llm_client.close_clients()
    await rerank_client.close_clients()
    await close_pool()
    log.info("shutdown_complete")


app = FastAPI(title="memory-service", version=__version__, lifespan=lifespan)

app.include_router(turns.router)
app.include_router(recall.router)
app.include_router(search.router)
app.include_router(memories.router)
app.include_router(cleanup.router)


@app.get("/health")
async def health() -> JSONResponse:
    settings = get_settings()
    db_ok = await ping()
    degraded: list[str] = []
    if not settings.llm_enabled:
        degraded.append("llm")
    if not settings.embed_enabled:
        degraded.append("embed")
    if not settings.rerank_enabled:
        degraded.append("rerank")

    body = {"status": "ok" if db_ok else "degraded", "version": __version__}
    if degraded:
        body["degraded"] = degraded
    code = 200 if db_ok else 503
    return JSONResponse(body, status_code=code)


# ── Resilience: §5 TASK.md "must not crash on malformed input" ─────────────
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logging.getLogger("memory.errors").exception("unhandled", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "see server logs"},
    )
