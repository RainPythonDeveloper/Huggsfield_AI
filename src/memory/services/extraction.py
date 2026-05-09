"""LLM-based fact extraction. Step 3 of the iteration plan.

Pipeline (per turn):
  1. Build prompt from messages.
  2. Call Alem `alemllm` chat with the extraction system prompt.
  3. Lenient-parse JSON (handles ```json fences and stray prose).
  4. Embed each memory's canonical "key: value" string.
  5. Insert into the `memories` table.

Failures are best-effort: if the LLM returns garbage, no memories are inserted
but the turn is still persisted (raw messages remain queryable in the
fallback path of Step 4 hybrid retrieval).
"""

import logging
from typing import Any

from memory import repository
from memory.clients import embeddings, llm
from memory.prompts import extract as ex_prompt
from memory.util.json_parse import parse_json_lenient

log = logging.getLogger(__name__)

VALID_TYPES = {"fact", "preference", "opinion", "event", "relation"}
MAX_VALUE_LEN = 500
MAX_KEY_LEN = 80


async def extract_and_store(
    *,
    turn_id: str,
    user_id: str | None,
    session_id: str,
    messages: list[dict[str, Any]],
) -> int:
    """Extract memories from a turn and persist them. Returns count inserted."""
    if not user_id:
        # No user → no place to attach durable memories. Skip extraction entirely;
        # the turn is still persisted and recallable session-scoped via raw msg fallback.
        return 0

    memories = await _llm_extract(messages)
    if not memories:
        return 0

    canonicals = [_canonical_text(m) for m in memories]
    vectors = await embeddings.embed_many(canonicals)

    inserted = 0
    for m, v in zip(memories, vectors, strict=True):
        try:
            await repository.insert_memory(
                user_id=user_id,
                session_id=session_id,
                type_=m["type"],
                key=m["key"],
                value=m["value"],
                confidence=float(m.get("confidence", 0.7)),
                raw_quote=m.get("raw_quote"),
                source_turn=turn_id,
                embedding_pgliteral=embeddings.to_pgvector(v),
            )
            inserted += 1
        except Exception as e:
            log.warning("memory_insert_failed", extra={"error": str(e), "key": m.get("key")})
    log.info(
        "extraction_complete",
        extra={
            "turn_id": turn_id,
            "user_id": user_id,
            "extracted": len(memories),
            "inserted": inserted,
        },
    )
    return inserted


def _canonical_text(m: dict[str, Any]) -> str:
    """Embed text. Joins key+value with a small natural-language template so the
    vector lives near typical user queries ("Where does the user work?")."""
    return f"User's {m['key'].replace('_', ' ')}: {m['value']}"


async def _llm_extract(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_prompt = ex_prompt.build_user_prompt(messages)
    try:
        raw = await llm.chat(system=ex_prompt.SYSTEM, user=user_prompt, temperature=0.0)
    except Exception as e:
        log.warning("llm_extract_call_failed", extra={"error": str(e)})
        return []
    parsed = parse_json_lenient(raw)
    if not parsed or not isinstance(parsed, dict):
        return []
    items = parsed.get("memories")
    if not isinstance(items, list):
        return []
    return [c for c in (_clean(m) for m in items) if c is not None]


def _clean(m: Any) -> dict[str, Any] | None:
    if not isinstance(m, dict):
        return None
    type_ = (m.get("type") or "").strip().lower()
    key = (m.get("key") or "").strip().lower()
    value = (m.get("value") or "").strip()
    if type_ not in VALID_TYPES or not key or not value:
        return None
    return {
        "type": type_,
        "key": key[:MAX_KEY_LEN],
        "value": value[:MAX_VALUE_LEN],
        "confidence": _coerce_conf(m.get("confidence")),
        "raw_quote": (m.get("raw_quote") or "")[:1000] or None,
    }


def _coerce_conf(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.7
    return max(0.0, min(1.0, f))
