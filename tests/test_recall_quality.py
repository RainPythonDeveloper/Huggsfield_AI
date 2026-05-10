"""Recall quality eval against fixtures/.

Reports recall@5 across all probes, plus breakdowns by category.
Run: pytest tests/test_recall_quality.py -s -v
"""

import json
from pathlib import Path

import httpx
import pytest
import yaml

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _ingest(client: httpx.Client, fixture_path: Path) -> str:
    data = json.loads(fixture_path.read_text())
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


@pytest.fixture(scope="module", autouse=True)
def _ingest_all_fixtures(client):
    for f in FIXTURES.glob("conv_*.json"):
        _ingest(client, f)
    yield
    for f in FIXTURES.glob("conv_*.json"):
        data = json.loads(f.read_text())
        client.delete(f"/users/{data['user_id']}")


def _load_probes() -> list[dict]:
    return yaml.safe_load((FIXTURES / "probes.yaml").read_text())["probes"]


def test_recall_quality(client: httpx.Client):
    probes = _load_probes()

    passed = 0
    failed_ids: list[tuple[str, str]] = []
    multi_hop_passed = 0
    multi_hop_total = 0
    noise_passed = 0
    noise_total = 0

    for p in probes:
        r = client.post(
            "/recall",
            json={
                "query": p["query"],
                "session_id": "probe-session",
                "user_id": p["user_id"],
                "max_tokens": 1024,
            },
        )
        assert r.status_code == 200, r.text
        ctx = r.json()["context"].lower()

        if p.get("must_be_empty"):
            ok = ctx.strip() == ""
            noise_total += 1
            if ok:
                noise_passed += 1
        else:
            need = [s.lower() for s in p.get("must_contain", [])]
            ok = all(s in ctx for s in need) if need else True

        if p.get("is_multi_hop"):
            multi_hop_total += 1
            if ok:
                multi_hop_passed += 1

        if ok:
            passed += 1
        else:
            failed_ids.append((p["id"], ctx[:200]))

    total = len(probes)
    recall_at_5 = passed / total
    print()
    print(f"=== RECALL QUALITY (Step 2 baseline) ===")
    print(f"overall recall@5: {passed}/{total} = {recall_at_5:.2%}")
    if multi_hop_total:
        print(f"multi-hop:        {multi_hop_passed}/{multi_hop_total} = {multi_hop_passed/multi_hop_total:.2%}")
    if noise_total:
        print(f"noise resistance: {noise_passed}/{noise_total} = {noise_passed/noise_total:.2%}")
    if failed_ids:
        print("--- failed probes ---")
        for pid, snippet in failed_ids:
            print(f"  ✗ {pid}: ctx={snippet!r}")

    # v1.0+ achieves 100% on this fixture. 0.85 leaves headroom for upstream
    # LLM/embed/rerank flakiness while still blocking real regressions.
    assert recall_at_5 >= 0.85, (
        f"recall@5 dropped to {recall_at_5:.2%} — investigate failed probes above"
    )
