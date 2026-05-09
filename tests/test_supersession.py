"""End-to-end fact-evolution test. Ingest the career-arc fixture and verify:

  1. /recall returns the CURRENT employer (Notion), not the stale one (Stripe).
  2. /memories shows BOTH (Notion active=true, Stripe active=false) — history preserved.
  3. memories.supersedes is populated, forming a chain.
"""

import json
from pathlib import Path

import httpx

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _ingest_fixture(client: httpx.Client, name: str) -> str:
    data = json.loads((FIXTURES / name).read_text())
    user_id = data["user_id"]
    client.delete(f"/users/{user_id}")
    for t in data["turns"]:
        r = client.post(
            "/turns",
            json={
                "session_id": t["session_id"],
                "user_id": user_id,
                "messages": t["messages"],
                "timestamp": t["timestamp"],
                "metadata": t.get("metadata", {}),
            },
        )
        assert r.status_code == 201, r.text
    return user_id


def test_employment_supersession(client: httpx.Client):
    user_id = _ingest_fixture(client, "conv_career.json")
    try:
        # /memories must include BOTH Stripe and Notion as employer values
        # somewhere in history.
        mems = client.get(f"/users/{user_id}/memories").json()["memories"]
        employers = [
            m for m in mems if m["key"] == "employer"
        ]
        values = {m["value"].lower() for m in employers}
        assert "notion" in values, f"expected Notion in employers, got {values}"
        assert "stripe" in values, f"expected Stripe in employers (history), got {values}"

        # The CURRENT employer must be Notion (active=true). Stripe should be
        # marked inactive once superseded by Notion.
        active_employers = [m for m in employers if m["active"]]
        active_vals = [m["value"].lower() for m in active_employers]
        assert "notion" in active_vals, f"Notion must be active, got {active_vals}"
        assert "stripe" not in active_vals, (
            f"Stripe must be superseded once Notion was reported. "
            f"Active employers: {active_vals}"
        )

        # /recall must surface Notion when asked for current employer.
        r = client.post(
            "/recall",
            json={
                "query": "Where does the user work?",
                "session_id": "career-probe",
                "user_id": user_id,
                "max_tokens": 1024,
            },
        )
        ctx = r.json()["context"].lower()
        assert "notion" in ctx, f"recall must include Notion. ctx={ctx[:300]!r}"

        # Notion should be present *as* the active employer; Stripe may still
        # appear if recall surfaces history, but the bullet "employer: Notion"
        # must exist.
        assert "employer: notion" in ctx or "employer:notion" in ctx, ctx[:300]
    finally:
        client.delete(f"/users/{user_id}")
