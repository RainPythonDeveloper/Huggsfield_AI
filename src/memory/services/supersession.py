"""Fact evolution + contradiction resolution.

Called from the extraction pipeline before INSERT. For each new candidate
that conflicts with existing active memories on the same `(user_id, key)`,
we ask an LLM judge for a verdict and act on it.

Failure mode: if the LLM is unavailable or returns garbage, we fall back to
a HEURISTIC for singular vs. multi-value keys — singular keys default to
`supersede` (newer wins), multi-value keys default to `coexist`. This keeps
ingest deterministic-enough even when the LLM is down.
"""

import logging
from typing import Any

from memory import repository
from memory.clients import llm
from memory.prompts import supersession as sup_prompt
from memory.util.json_parse import parse_json_lenient

log = logging.getLogger(__name__)

VALID_VERDICTS = {"supersede", "coexist", "keep_old", "noop"}

# Keys for which it's normal to have several active values at once. Used as
# a heuristic fallback when the LLM judge is unavailable.
MULTI_VALUE_KEYS = {
    "pet_dog_name",
    "pet_cat_name",
    "pet_name",
    "child_name",
    "language_spoken",
    "hobby",
    "friend_name",
    "favorite_book",
    "favorite_movie",
    "favorite_food",
    "currently_reading",
}


async def resolve(
    *,
    user_id: str,
    key: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Decide what to do with a new candidate vs. existing active memories.

    Returns:
        {
            "verdict": "supersede" | "coexist" | "keep_old" | "noop",
            "supersede_ids": list[uuid str] | [],   # set when verdict=supersede
            "reason": str,
        }
    """
    existing = await repository.find_active_memories_by_key(user_id=user_id, key=key)
    if not existing:
        return {"verdict": "coexist", "supersede_ids": [], "reason": "no existing"}

    # Exact same value → duplicate, skip.
    if any(_eq_value(e["value"], candidate["value"]) for e in existing):
        return {"verdict": "noop", "supersede_ids": [], "reason": "duplicate value"}

    verdict, reason = await _judge(key=key, existing=existing, candidate=candidate)
    supersede_ids: list[str] = []
    if verdict == "supersede":
        supersede_ids = [e["id"] for e in existing]
    return {"verdict": verdict, "supersede_ids": supersede_ids, "reason": reason}


async def _judge(
    *,
    key: str,
    existing: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> tuple[str, str]:
    """LLM judge with deterministic heuristic fallback."""
    try:
        raw = await llm.chat(
            system=sup_prompt.SYSTEM,
            user=sup_prompt.build_user_prompt(
                key=key, existing=existing, candidate=candidate
            ),
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as e:
        log.warning("supersession_llm_failed_using_heuristic", extra={"error": str(e), "key": key})
        return _heuristic_verdict(key), "llm_unavailable"

    parsed = parse_json_lenient(raw)
    if not isinstance(parsed, dict):
        log.warning("supersession_judge_unparseable", extra={"sample": raw[:200]})
        return _heuristic_verdict(key), "unparseable"

    verdict = (parsed.get("verdict") or "").strip().lower()
    if verdict not in VALID_VERDICTS:
        return _heuristic_verdict(key), f"bad_verdict={verdict!r}"
    return verdict, str(parsed.get("reason") or "")


def _heuristic_verdict(key: str) -> str:
    """Fallback when LLM unavailable. Singular keys → supersede, plural → coexist."""
    return "coexist" if key in MULTI_VALUE_KEYS else "supersede"


def _eq_value(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()
