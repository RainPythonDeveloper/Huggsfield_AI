"""Contract roundtrip tests against a running container."""

import httpx


def test_health(client: httpx.Client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")


def test_turn_roundtrip(client: httpx.Client):
    # cleanup first
    client.delete("/users/contract-u1")

    r = client.post(
        "/turns",
        json={
            "session_id": "contract-s1",
            "user_id": "contract-u1",
            "messages": [
                {"role": "user", "content": "I just moved to Berlin from NYC last month."},
                {"role": "assistant", "content": "Berlin is great. How are you settling in?"},
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    )
    assert r.status_code == 201
    assert "id" in r.json()

    r = client.post(
        "/recall",
        json={
            "query": "Where does this user live?",
            "session_id": "contract-s1",
            "user_id": "contract-u1",
            "max_tokens": 256,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "context" in body
    assert "citations" in body
    assert isinstance(body["citations"], list)

    r = client.get("/users/contract-u1/memories")
    assert r.status_code == 200
    assert "memories" in r.json()

    # cleanup
    r = client.delete("/users/contract-u1")
    assert r.status_code == 204


def test_cold_session(client: httpx.Client):
    r = client.post(
        "/recall",
        json={
            "query": "anything",
            "session_id": "this-session-does-not-exist",
            "user_id": "ghost",
            "max_tokens": 256,
        },
    )
    assert r.status_code == 200
    assert r.json()["context"] == ""
    assert r.json()["citations"] == []


def test_malformed_json(client: httpx.Client):
    r = client.post(
        "/turns",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


def test_missing_required_fields(client: httpx.Client):
    r = client.post("/turns", json={"messages": []})
    assert r.status_code == 422


def test_unicode_payload(client: httpx.Client):
    client.delete("/users/uni-u1")
    r = client.post(
        "/turns",
        json={
            "session_id": "uni-s1",
            "user_id": "uni-u1",
            "messages": [
                {"role": "user", "content": "Привет 🐶 Я живу в Берлине"},
                {"role": "assistant", "content": "Здорово!"},
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {"locale": "ru-RU"},
        },
    )
    assert r.status_code == 201
    client.delete("/users/uni-u1")


def test_concurrent_sessions_no_bleed(client: httpx.Client):
    # Two distinct users — recall for one must not see the other's facts.
    client.delete("/users/iso-a")
    client.delete("/users/iso-b")

    client.post(
        "/turns",
        json={
            "session_id": "iso-sa",
            "user_id": "iso-a",
            "messages": [
                {"role": "user", "content": "I work at Apple in Cupertino."},
                {"role": "assistant", "content": "Great."},
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    )
    client.post(
        "/turns",
        json={
            "session_id": "iso-sb",
            "user_id": "iso-b",
            "messages": [
                {"role": "user", "content": "I work at Microsoft in Redmond."},
                {"role": "assistant", "content": "Great."},
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    )

    r = client.post(
        "/recall",
        json={"query": "Where does the user work?", "session_id": "iso-sa", "user_id": "iso-a", "max_tokens": 256},
    )
    text = r.json()["context"].lower()
    assert "apple" in text or "cupertino" in text
    assert "microsoft" not in text and "redmond" not in text

    client.delete("/users/iso-a")
    client.delete("/users/iso-b")
