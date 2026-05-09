"""Robust JSON extraction from LLM completions.

Alem `alemllm` wraps JSON in ```json ... ``` fences (verified by curl probe).
Some prompts also produce a leading prose intro before the JSON. We try in
order of strictness:

  1. Direct json.loads on the whole string.
  2. Strip a markdown fence (```json ... ``` or ``` ... ```).
  3. Find the first balanced {...} block by bracket counting and parse that.

Returns None on total failure rather than raising — extraction is best-effort
and a single malformed reply must not break a /turns ingest.
"""

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_json_lenient(raw: str) -> Any | None:
    if not raw:
        return None
    s = raw.strip()

    # 1. straight parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2. ```json fence
    m = _FENCE_RE.search(s)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3. first balanced {} block
    block = _first_balanced_object(s)
    if block:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass

    log.warning("json_parse_failed", extra={"sample": s[:300]})
    return None


def _first_balanced_object(s: str) -> str | None:
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None
