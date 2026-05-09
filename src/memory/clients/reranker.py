"""Alem reranker client (`/v1/rerank`, Cohere-compatible).

Probe response (verified by curl):
    {
      "id": "...",
      "results": [
        {"index": 2, "relevance_score": 0.9995, "document": {"text": "..."}},
        ...
      ],
      "meta": {...}
    }
"""

import logging

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
            base_url=s.rerank_base_url,
            headers={"Authorization": f"Bearer {s.rerank_api_key}"},
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
async def rerank(
    *,
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[dict]:
    """Return [{"index": i, "score": s}, ...] ordered best-first.

    `index` refers to the position in the input `documents` list. Caller is
    expected to map back to the original record. We don't return the document
    text — caller already has it.
    """
    s = get_settings()
    if not s.rerank_enabled:
        raise RuntimeError("reranker disabled (no RERANK_API_KEY)")
    if not documents:
        return []

    payload = {
        "model": s.rerank_model,
        "query": query,
        "documents": documents,
    }
    if top_n is not None:
        payload["top_n"] = top_n

    r = await _get_client().post("/rerank", json=payload)
    r.raise_for_status()
    data = r.json()
    return [
        {"index": int(item["index"]), "score": float(item["relevance_score"])}
        for item in data.get("results", [])
    ]
