"""Alem embeddings client (OpenAI-compatible). Returns vectors of dim=1024."""

import asyncio
import logging
from collections.abc import Sequence

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from memory.config import get_settings

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        s = get_settings()
        _client = httpx.AsyncClient(
            base_url=s.embed_base_url,
            headers={"Authorization": f"Bearer {s.embed_api_key}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
            http2=True,
        )
    return _client


async def close_clients() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
async def _embed_one(text: str) -> list[float]:
    s = get_settings()
    if not s.embed_enabled:
        raise RuntimeError("embeddings disabled (no EMBED_API_KEY)")
    r = await _get_client().post(
        "/embeddings",
        json={"model": s.embed_model, "input": text},
    )
    r.raise_for_status()
    data = r.json()
    return data["data"][0]["embedding"]


async def embed(text: str) -> list[float]:
    """Embed a single text. Truncates extremely long inputs at 8000 chars."""
    text = text[:8000] if len(text) > 8000 else text
    return await _embed_one(text)


async def embed_many(texts: Sequence[str]) -> list[list[float]]:
    """Concurrent embedding of multiple texts (Alem endpoint takes one input at a time)."""
    if not texts:
        return []
    return await asyncio.gather(*(embed(t) for t in texts))


def to_pgvector(vec: list[float]) -> str:
    """Format a Python list as pgvector literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
