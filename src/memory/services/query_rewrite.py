"""Query analysis + decomposition for multi-hop recall."""

import logging

from memory.clients import llm
from memory.prompts import query_rewrite as qr_prompt
from memory.util.json_parse import parse_json_lenient

log = logging.getLogger(__name__)

MAX_SUB_QUERIES = 3


async def analyze(query: str) -> dict:
    """Return {"is_multi_hop": bool, "sub_queries": list[str]}.

    LLM failure → return single-hop default ({"is_multi_hop": False, "sub_queries": []}),
    so recall keeps working on the original query.
    """
    default = {"is_multi_hop": False, "sub_queries": []}
    try:
        raw = await llm.chat(
            system=qr_prompt.SYSTEM,
            user=qr_prompt.build_user_prompt(query),
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as e:
        log.warning("query_rewrite_llm_failed", extra={"error": str(e)})
        return default

    parsed = parse_json_lenient(raw)
    if not isinstance(parsed, dict):
        return default

    is_mh = bool(parsed.get("is_multi_hop"))
    raw_subs = parsed.get("sub_queries") or []
    if not isinstance(raw_subs, list):
        return default
    subs = [s.strip() for s in raw_subs if isinstance(s, str) and s.strip()]
    subs = subs[:MAX_SUB_QUERIES]

    if is_mh and not subs:
        # Said multi-hop but gave nothing usable — degrade.
        return default
    return {"is_multi_hop": is_mh, "sub_queries": subs}
