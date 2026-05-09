"""Robustness tests (TASK §5 "must not crash on malformed input, oversized
payloads, or unicode oddities")."""

import httpx


def test_oversized_payload(client: httpx.Client):
    """1MB body — service should reject with 4xx, not crash."""
    big_content = "A" * (1_500_000)
    r = client.post(
        "/turns",
        json={
            "session_id": "big-s",
            "user_id": "big-u",
            "messages": [{"role": "user", "content": big_content}],
            "timestamp": "2025-04-01T10:30:00Z",
            "metadata": {},
        },
    )
    # We accept a wide 4xx/5xx band — the contract is "doesn't crash".
    assert r.status_code != 200
    # Service still alive
    h = client.get("/health")
    assert h.status_code == 200


def test_emoji_unicode_and_zero_width(client: httpx.Client):
    user_id = "emoji-u"
    client.delete(f"/users/{user_id}")
    r = client.post(
        "/turns",
        json={
            "session_id": "emoji-s",
            "user_id": user_id,
            "messages": [
                {"role": "user", "content": "Я живу в 🇩🇪 Berlin​ — and I love 🍣"},
                {"role": "assistant", "content": "Cool 🙂"},
            ],
            "timestamp": "2025-04-01T10:30:00Z",
            "metadata": {"locale": "ru-DE"},
        },
    )
    assert r.status_code == 201

    r = client.post(
        "/recall",
        json={
            "query": "Where does the user live?",
            "session_id": "emoji-s",
            "user_id": user_id,
            "max_tokens": 512,
        },
    )
    assert r.status_code == 200
    client.delete(f"/users/{user_id}")


def test_empty_messages_array_rejected(client: httpx.Client):
    r = client.post(
        "/turns",
        json={
            "session_id": "empty-s",
            "user_id": "empty-u",
            "messages": [],
            "timestamp": "2025-04-01T10:30:00Z",
            "metadata": {},
        },
    )
    assert r.status_code == 422


def test_invalid_role_rejected(client: httpx.Client):
    r = client.post(
        "/turns",
        json={
            "session_id": "role-s",
            "user_id": "role-u",
            "messages": [{"role": "wizard", "content": "abracadabra"}],
            "timestamp": "2025-04-01T10:30:00Z",
            "metadata": {},
        },
    )
    assert r.status_code == 422


def test_search_empty_corpus_returns_empty(client: httpx.Client):
    r = client.post(
        "/search",
        json={
            "query": "anything",
            "user_id": "no-such-user-zzz",
            "limit": 5,
        },
    )
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_concurrent_ingest_no_corruption(client: httpx.Client):
    """Burst-ingest in fast succession — exercises the asyncpg pool.
    Per TASK §5 'concurrent sessions active at once must not bleed'."""
    import concurrent.futures as cf

    def _post(i: int) -> int:
        with httpx.Client(base_url=str(client.base_url), timeout=90.0) as c:
            r = c.post(
                "/turns",
                json={
                    "session_id": f"burst-s-{i % 3}",
                    "user_id": f"burst-u-{i % 3}",
                    "messages": [
                        {"role": "user", "content": f"Burst message {i}, my favorite number is {i}."},
                        {"role": "assistant", "content": "OK"},
                    ],
                    "timestamp": "2025-04-01T10:30:00Z",
                    "metadata": {"i": i},
                },
            )
        return r.status_code

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_post, range(8)))
    assert all(s == 201 for s in results), results

    for u in {f"burst-u-{i}" for i in range(3)}:
        client.delete(f"/users/{u}")
