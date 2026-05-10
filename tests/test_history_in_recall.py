"""History-surfaces-in-/recall test.

Closes the gap that `test_employment_supersession` only checks accidentally:
in the career fixture, the Notion turn says "Stripe was great but I needed a
change", which puts "Stripe" into the new memory's `raw_quote`. So /recall
returned Stripe even when only the *active* Notion memory was retrieved.

This test states the new job *without* referencing the old one. After the v1.1
history-aware recall, /recall must still surface the superseded "Stripe" fact
when the user asks a history-shaped question — per TASK §3 example output
("Works at Notion as a PM ... previously at Stripe ...") and §9.A
("Does it still know the history?").
"""

import httpx


def test_history_recall_without_quote_overlap(client: httpx.Client):
    user = "hist-clean"
    client.delete(f"/users/{user}")
    try:
        # Session 1: only mentions Stripe.
        r = client.post(
            "/turns",
            json={
                "session_id": "h-s1",
                "user_id": user,
                "messages": [
                    {"role": "user", "content": "I work at Stripe as a backend engineer."},
                    {"role": "assistant", "content": "Got it."},
                ],
                "timestamp": "2025-01-10T09:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text

        # Session 2: only mentions Notion. NO mention of Stripe in user content.
        r = client.post(
            "/turns",
            json={
                "session_id": "h-s2",
                "user_id": user,
                "messages": [
                    {"role": "user", "content": "Quick update — I just joined Notion as a product manager."},
                    {"role": "assistant", "content": "Congrats on the move."},
                ],
                "timestamp": "2025-04-01T10:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text

        # /memories must show both employers (history preserved).
        mems = client.get(f"/users/{user}/memories").json()["memories"]
        employers = {m["value"].lower(): m["active"] for m in mems if m["key"] == "employer"}
        assert "notion" in employers, f"expected Notion, got {employers}"
        assert "stripe" in employers, f"expected Stripe in history, got {employers}"
        assert employers["notion"] is True, "Notion must be active"
        assert employers["stripe"] is False, "Stripe must be superseded"

        # /recall on a history-shaped question must surface Stripe.
        r = client.post(
            "/recall",
            json={
                "query": "Has the user ever worked at Stripe?",
                "session_id": "h-probe",
                "user_id": user,
                "max_tokens": 1024,
            },
        )
        assert r.status_code == 200
        ctx = r.json()["context"].lower()
        assert "stripe" in ctx, f"history must surface in /recall. ctx={ctx[:400]!r}"

        # /recall on a present-tense question still prefers Notion.
        r = client.post(
            "/recall",
            json={
                "query": "Where does the user work today?",
                "session_id": "h-probe",
                "user_id": user,
                "max_tokens": 1024,
            },
        )
        ctx2 = r.json()["context"].lower()
        assert "notion" in ctx2, f"current employer must dominate. ctx={ctx2[:400]!r}"

        # And we can find the (historical) marker on the inactive bullet so a
        # frozen LLM can disambiguate. Either a "previously"/"historical" tag
        # OR Notion appears strictly before Stripe (Bucket 1 vs Bucket 2).
        if "stripe" in ctx2:
            n_pos = ctx2.index("notion")
            s_pos = ctx2.index("stripe")
            assert n_pos < s_pos or "(historical)" in ctx2, (
                f"current Notion must precede or be marked vs. historical Stripe. ctx={ctx2[:400]!r}"
            )
    finally:
        client.delete(f"/users/{user}")
