"""Alem LLM client (chat completions, OpenAI-compatible)."""

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
            base_url=s.alem_llm_base_url,
            headers={"Authorization": f"Bearer {s.alem_api_key}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
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
async def chat(
    *,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Single-turn chat completion. Returns the raw assistant text."""
    s = get_settings()
    if not s.llm_enabled:
        raise RuntimeError("llm disabled (no ALEM_API_KEY)")

    payload: dict = {
        "model": s.alem_llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    r = await _get_client().post("/chat/completions", json=payload)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"] or ""
