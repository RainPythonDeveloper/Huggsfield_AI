"""Restart-survives-data test (TASK §5 hard constraint).

Restarts the app container (NOT the db — db data lives on the named volume
either way) and checks the memory comes back. Skipped when SKIP_RESTART_TESTS
is set (e.g., on CI without docker).
"""

import os
import subprocess
import time

import httpx
import pytest


@pytest.mark.skipif(
    os.environ.get("SKIP_RESTART_TESTS") == "1",
    reason="restart tests require docker compose",
)
def test_restart_persistence(client: httpx.Client):
    user_id = "persist-u"
    client.delete(f"/users/{user_id}")

    r = client.post(
        "/turns",
        json={
            "session_id": "persist-s",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I work at Vercel as a staff engineer.",
                },
                {"role": "assistant", "content": "Cool"},
            ],
            "timestamp": "2025-04-01T10:30:00Z",
            "metadata": {},
        },
    )
    assert r.status_code == 201
    pre_mems = client.get(f"/users/{user_id}/memories").json()["memories"]
    assert len(pre_mems) > 0, "should have extracted at least one memory"

    # Restart BOTH containers — exercises the docker volume not just the app process.
    subprocess.run(
        ["docker", "compose", "restart"],
        check=True,
        capture_output=True,
    )

    # Wait for /health to return 200.
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            h = client.get("/health", timeout=2.0)
            if h.status_code == 200:
                break
        except httpx.RequestError:
            pass
        time.sleep(1)
    else:
        pytest.fail("service did not come back healthy within 60s")

    post_mems = client.get(f"/users/{user_id}/memories").json()["memories"]
    pre_keys = {(m["key"], m["value"]) for m in pre_mems}
    post_keys = {(m["key"], m["value"]) for m in post_mems}
    assert pre_keys == post_keys, (
        f"memories not preserved across restart: lost={pre_keys - post_keys}, "
        f"new={post_keys - pre_keys}"
    )

    # /recall should still find the persisted fact.
    r = client.post(
        "/recall",
        json={
            "query": "Where does the user work?",
            "session_id": "persist-s",
            "user_id": user_id,
            "max_tokens": 512,
        },
    )
    assert r.status_code == 200
    assert "vercel" in r.json()["context"].lower()

    client.delete(f"/users/{user_id}")
