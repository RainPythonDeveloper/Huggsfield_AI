"""Token-budget compliance test.

Per TASK.md §3 *"Should respect max_tokens (approximate is fine; don't blow
past it by 2x)"* and §3 priority rule *"stable user facts first, then
query-relevant memories, then recent context"*.
"""

import json
from pathlib import Path

import httpx
import pytest
import tiktoken

FIXTURES = Path(__file__).parent.parent / "fixtures"
_enc = tiktoken.get_encoding("cl100k_base")


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
        assert r.status_code == 201
    return user_id


@pytest.fixture(scope="module", autouse=True)
def _ingest(client):
    user_id = _ingest_fixture(client, "conv_pets.json")
    yield user_id
    client.delete(f"/users/{user_id}")


@pytest.mark.parametrize("budget", [128, 256, 512, 1024])
def test_budget_respected(client: httpx.Client, budget: int, _ingest):
    user_id = _ingest
    r = client.post(
        "/recall",
        json={
            "query": "Tell me about the user's pets and their daily routine.",
            "session_id": "budget-probe",
            "user_id": user_id,
            "max_tokens": budget,
        },
    )
    assert r.status_code == 200
    ctx = r.json()["context"]
    actual = len(_enc.encode(ctx))
    # Per TASK §3: don't blow past 2x. We aim for 0.95x as our soft cap.
    assert actual <= budget * 1.10, (
        f"budget={budget}, actual={actual}, ctx={ctx[:200]!r}"
    )


def test_user_facts_priority_at_tight_budget(client: httpx.Client, _ingest):
    """At tight budgets, the user's stable facts (pet name) must win over
    everything else."""
    user_id = _ingest
    r = client.post(
        "/recall",
        json={
            "query": "What does the user have?",
            "session_id": "budget-probe",
            "user_id": user_id,
            "max_tokens": 128,
        },
    )
    ctx = r.json()["context"].lower()
    # At 128 tokens we expect the most stable user fact ("pet dog name: Biscuit")
    # to be the first thing we see.
    assert "biscuit" in ctx, f"pet name must survive tight budget. ctx={ctx!r}"
