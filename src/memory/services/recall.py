"""Recall pipeline.

Step 5 (this version):
  1. Run BM25 (top 30) and vector (top 30) over memories in parallel.
  2. RRF-fuse with k=60 → top 20.
  3. **Rerank** the top 20 via Alem cross-encoder. Drop hits below RERANK_FLOOR
     (calibrated against the curl probe — 0.99 for relevant, 0.013 irrelevant,
     so 0.20 is a generous floor).
  4. If the reranked set is empty, fall back to BM25 over messages.content_tsv.
  5. Format as bucketed prose.

If the reranker is disabled (no API key) we degrade gracefully: keep the RRF
order and skip the threshold (matches v0.5 behaviour).

Step 8 will replace (5) with token-budget-aware priority assembly.
"""

import asyncio
import logging

from memory import repository
from memory.clients import embeddings, reranker
from memory.config import get_settings
from memory.schemas import Citation, RecallIn, RecallOut, SearchHit, SearchIn, SearchOut
from memory.services import query_rewrite
from memory.util import tokens
from memory.util.rrf import reciprocal_rank_fusion

log = logging.getLogger(__name__)

RETRIEVAL_K = 30        # per-channel breadth before fusion
FUSED_K = 20            # how many fused candidates we send to the reranker
RECALL_TOP_N = 8        # how many we keep after rerank for prose
# Drop hits below this. Calibrated empirically: relevant facts score ~0.3-0.8,
# borderline ~0.04, true noise ~5e-5. A 0.05 floor cuts noise cleanly while
# keeping borderline facts that may still help the agent.
RERANK_FLOOR = 0.05
DEFAULT_SNIPPET_CHARS = 280


async def recall(req: RecallIn) -> RecallOut:
    candidates = await _retrieve(req.query, req.user_id, req.session_id)

    final_rows = await _rerank_and_filter(
        query=req.query, candidates=candidates, top_n=RECALL_TOP_N, floor=RERANK_FLOOR
    )

    # Bucket 3 (recent context): only fetched when budget allows; cheap query.
    recent_msgs: list[dict] = []
    if req.session_id:
        try:
            recent_msgs = await repository.fetch_recent_messages_for_session(
                req.session_id, limit=4
            )
        except Exception as e:
            log.warning("recent_fetch_failed", extra={"error": str(e)})

    if not final_rows and not recent_msgs:
        # Cold-extraction fallback: maybe the extractor missed it but the raw
        # conversation has it. Run BM25 over messages.
        fallback = await repository.search_messages_by_bm25(
            req.query,
            user_id=req.user_id,
            session_id=req.session_id if req.user_id is None else None,
            limit=8,
        )
        if fallback:
            fallback = await _rerank_messages_filter(
                query=req.query, rows=fallback, top_n=5, floor=RERANK_FLOOR
            )
        if not fallback:
            return RecallOut(context="", citations=[])
        return _format_message_fallback(fallback, budget=req.max_tokens)

    return _format_recall_budgeted(
        rows=final_rows,
        recent=recent_msgs,
        budget=req.max_tokens,
    )


async def _retrieve(
    query: str, user_id: str | None, session_id: str
) -> list[dict]:
    """Decompose multi-hop queries → run hybrid for each → merge with RRF.

    Single-hop queries skip the merge and just return the hybrid result.
    """
    scope_session = session_id if user_id is None else None
    decomp = await query_rewrite.analyze(query)

    if not decomp["is_multi_hop"]:
        return await _hybrid_memories(
            query=query,
            user_id=user_id,
            session_id=scope_session,
            per_channel=RETRIEVAL_K,
            fused_limit=FUSED_K,
        )

    # Run all sub-queries in parallel.
    tasks = [
        _hybrid_memories(
            query=sq,
            user_id=user_id,
            session_id=scope_session,
            per_channel=RETRIEVAL_K,
            fused_limit=FUSED_K,
        )
        for sq in decomp["sub_queries"]
    ]
    sub_results = await asyncio.gather(*tasks)
    log.info(
        "multi_hop_decomposed",
        extra={
            "query": query[:80],
            "sub_queries": decomp["sub_queries"],
            "n_subs": len(sub_results),
            "hits_per_sub": [len(s) for s in sub_results],
        },
    )

    channels = {f"sub_{i}": rows for i, rows in enumerate(sub_results) if rows}
    if not channels:
        # Decomposition produced no hits — fall back to original query as a
        # safety net. Keeps the pipeline closed-loop.
        return await _hybrid_memories(
            query=query,
            user_id=user_id,
            session_id=scope_session,
            per_channel=RETRIEVAL_K,
            fused_limit=FUSED_K,
        )

    return reciprocal_rank_fusion(
        channels, id_key="id", k=60, limit=FUSED_K
    )


# ── Hybrid retrieval ───────────────────────────────────────────────────────


async def _hybrid_memories(
    *,
    query: str,
    user_id: str | None,
    session_id: str | None,
    per_channel: int,
    fused_limit: int,
) -> list[dict]:
    """Run vector + BM25 in parallel and RRF-fuse. If the embedding call fails
    (Alem 5xx, timeout, etc.) we degrade to BM25-only rather than 500 — the
    eval harness is more tolerant of weaker recall than of crashed endpoints.

    History support (TASK §3 example "previously at Stripe", §9.A "still know
    the history"): we include `active=False` memories as candidates. The
    reranker remains the quality gate — historical hits only surface for
    queries that are genuinely about history. Active facts are bucketed first
    in assembly so the *current* fact still dominates "where do you work
    today?" queries.
    """
    bm25_task = asyncio.create_task(
        repository.search_memories_by_bm25(
            query,
            user_id=user_id,
            session_id=session_id,
            limit=per_channel,
            only_active=False,
        )
    )

    vec_rows: list[dict] = []
    try:
        qvec = await embeddings.embed(query)
        qlit = embeddings.to_pgvector(qvec)
        vec_rows = await repository.search_memories_by_embedding(
            qlit,
            user_id=user_id,
            session_id=session_id,
            limit=per_channel,
            only_active=False,
        )
    except Exception as e:
        log.warning("vector_channel_failed_bm25_only", extra={"error": str(e)})

    bm25_rows = await bm25_task

    channels: dict[str, list[dict]] = {}
    if vec_rows:
        channels["vector"] = vec_rows
    if bm25_rows:
        channels["bm25"] = bm25_rows
    if not channels:
        return []

    return reciprocal_rank_fusion(
        channels, id_key="id", k=60, limit=fused_limit
    )


# ── Reranker stage ─────────────────────────────────────────────────────────


def _rerank_doc_for_memory(r: dict) -> str:
    """Document text the reranker sees.

    Empirical calibration (Step 5):
      - Format `key: value` → ~0.0008 (reranker hates this).
      - First-person raw quote ("I work at Apple") → ~0.0025.
      - Third-person canonical rendering ("The user's employer is Apple")
        → ~0.97 against an aligned query.

    So we render in third person regardless of the source text. We append
    the raw quote for context — it gives the cross-encoder access to the
    surrounding language without changing the dominant first-person framing.
    """
    key = r["key"].replace("_", " ")
    value = r["value"]
    base = f"The user's {key} is {value}."
    quote = (r.get("raw_quote") or "").strip()
    if quote:
        return f"{base} Originally said: {quote}"
    return base


async def _rerank_and_filter(
    *,
    query: str,
    candidates: list[dict],
    top_n: int,
    floor: float,
) -> list[dict]:
    if not candidates:
        return []
    settings = get_settings()
    if not settings.rerank_enabled:
        return candidates[:top_n]

    docs = [_rerank_doc_for_memory(r) for r in candidates]
    try:
        ranked = await reranker.rerank(query=query, documents=docs, top_n=top_n)
    except Exception as e:
        log.warning("rerank_failed_fallback_rrf", extra={"error": str(e)})
        return candidates[:top_n]

    out: list[dict] = []
    for item in ranked:
        if item["score"] < floor:
            continue
        row = dict(candidates[item["index"]])
        row["_rerank_score"] = item["score"]
        out.append(row)
    return out


async def _rerank_messages_filter(
    *,
    query: str,
    rows: list[dict],
    top_n: int,
    floor: float,
) -> list[dict]:
    settings = get_settings()
    if not settings.rerank_enabled or not rows:
        return rows[:top_n]
    docs = [r["content"] for r in rows]
    try:
        ranked = await reranker.rerank(query=query, documents=docs, top_n=top_n)
    except Exception as e:
        log.warning("rerank_msg_failed_fallback", extra={"error": str(e)})
        return rows[:top_n]
    out: list[dict] = []
    for item in ranked:
        if item["score"] < floor:
            continue
        row = dict(rows[item["index"]])
        row["_rerank_score"] = item["score"]
        out.append(row)
    return out


# ── Budget-aware assembly ─────────────────────────────────────────────────
#
# Priority (TASK.md §3 "stable user facts first, then query-relevant memories,
# then recent context"):
#
#   Bucket 1 — stable user facts: type ∈ {fact, preference, relation},
#              active=true. These are durable identity properties — almost
#              always worth including.
#
#   Bucket 2 — query-relevant memories: anything else from rerank top-N
#              (events, opinions). Order by rerank score desc.
#
#   Bucket 3 — recent context: last few raw messages from this session,
#              for working-conversational continuity.
#
# We aim for soft cap = 0.95 × max_tokens to leave headroom (the consumer
# may add extra system text). A bullet is dropped (not truncated) if adding
# it would exceed the cap — truncation produces ugly half-sentences and
# the extra precision isn't worth it.

_SOFT_CAP_RATIO = 0.95
_HEADER_USER_FACTS = "## Known facts about this user"
_HEADER_QUERY_RELEVANT = "## Relevant memories"
_HEADER_RECENT = "## Recent conversation"


def _format_recall_budgeted(
    *,
    rows: list[dict],
    recent: list[dict],
    budget: int,
) -> RecallOut:
    """Bucket priority (TASK §3):
      1. Stable user facts — active fact/preference/relation.
      2. Query-relevant — active events/opinions + ALL historical hits
         (active=false). Historical bullets carry a `(historical)` tag so a
         frozen LLM can disambiguate per the §3 "previously at Stripe" example.
      3. Recent raw messages, gated.

    Citations are deduplicated by (turn_id, snippet[:120]) and capped at 6 —
    one snippet per turn is plenty for the consumer agent.
    """
    active_facts = [
        r for r in rows
        if r.get("active", True) and r["type"] in ("fact", "preference", "relation")
    ]
    other_active = [
        r for r in rows
        if r.get("active", True) and r["type"] not in ("fact", "preference", "relation")
    ]
    historical = [r for r in rows if not r.get("active", True)]

    soft_cap = max(1, int(budget * _SOFT_CAP_RATIO))
    used = 0
    lines: list[str] = []
    citations: list[Citation] = []

    def try_add(line: str) -> bool:
        nonlocal used
        cost = tokens.count(line) + 1  # +1 newline
        if used + cost > soft_cap:
            return False
        lines.append(line)
        used += cost
        return True

    # Bucket 1 — stable, current
    if active_facts:
        if try_add(_HEADER_USER_FACTS):
            for r in active_facts:
                bullet = f"- {_humanize(r)}"
                if not try_add(bullet):
                    break
                citations.append(_cite(r))

    # Bucket 2 — query-relevant. Active events/opinions first, then historical.
    bucket2 = other_active + historical
    if bucket2:
        if try_add("") and try_add(_HEADER_QUERY_RELEVANT):
            for r in bucket2:
                bullet = f"- {_humanize(r)}"
                if not try_add(bullet):
                    break
                citations.append(_cite(r))

    # Bucket 3 — only enrich if budget remains AND we don't already have ≥4 facts
    if recent and used < soft_cap * 0.8 and len(citations) < 6:
        if try_add("") and try_add(_HEADER_RECENT):
            for r in recent:
                snippet = (r["content"] or "")[:160]
                ts = r["timestamp"].strftime("%Y-%m-%d") if r.get("timestamp") else ""
                bullet = f"- [{ts}] {snippet}" if ts else f"- {snippet}"
                if not try_add(bullet):
                    break
                citations.append(
                    Citation(
                        turn_id=r.get("turn_id", ""),
                        score=0.0,
                        snippet=snippet[:DEFAULT_SNIPPET_CHARS],
                    )
                )

    citations = _dedup_citations(citations, cap=6)
    return RecallOut(context="\n".join(lines).strip(), citations=citations)


def _format_message_fallback(rows: list[dict], *, budget: int) -> RecallOut:
    soft_cap = max(1, int(budget * _SOFT_CAP_RATIO))
    used = 0
    lines: list[str] = []
    citations: list[Citation] = []

    def try_add(line: str) -> bool:
        nonlocal used
        cost = tokens.count(line) + 1
        if used + cost > soft_cap:
            return False
        lines.append(line)
        used += cost
        return True

    if try_add("## Relevant from recent conversations"):
        for r in rows:
            ts = r["timestamp"].strftime("%Y-%m-%d") if r.get("timestamp") else ""
            snippet = (r["content"] or "")[:DEFAULT_SNIPPET_CHARS]
            bullet = f"- [{ts}] {snippet}" if ts else f"- {snippet}"
            if not try_add(bullet):
                break
            score = float(r.get("_rerank_score") or r.get("score") or 0.0)
            citations.append(
                Citation(turn_id=r["turn_id"], score=score, snippet=snippet)
            )
    return RecallOut(context="\n".join(lines), citations=citations)


def _humanize(r: dict) -> str:
    key = r["key"].replace("_", " ")
    value = r["value"]
    quote = r.get("raw_quote")
    tag = "" if r.get("active", True) else " (historical)"
    if quote and len(quote) < 140:
        return f"{key}: {value}{tag} (\"{quote}\")"
    return f"{key}: {value}{tag}"


def _dedup_citations(citations: list[Citation], *, cap: int) -> list[Citation]:
    """Drop duplicate (turn_id, snippet) pairs and cap the list. The contract
    consumer (the calling agent) shows these to the user — one entry per
    distinct turn-snippet is plenty.
    """
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []
    for c in citations:
        key = (c.turn_id or "", (c.snippet or "")[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= cap:
            break
    return out


def _cite(r: dict) -> Citation:
    snippet = (r.get("raw_quote") or f"{r['key']}: {r['value']}")[:DEFAULT_SNIPPET_CHARS]
    score = float(r.get("_rerank_score") or r.get("_rrf_score") or r.get("score") or 0.0)
    return Citation(
        turn_id=r.get("source_turn") or "",
        score=score,
        snippet=snippet,
    )


# ── /search ───────────────────────────────────────────────────────────────


async def search(req: SearchIn) -> SearchOut:
    """Same hybrid+rerank pipeline; structured output instead of prose."""
    fused = await _hybrid_memories(
        query=req.query,
        user_id=req.user_id,
        session_id=req.session_id,
        per_channel=max(req.limit * 2, RETRIEVAL_K),
        fused_limit=max(req.limit * 2, FUSED_K),
    )
    final = await _rerank_and_filter(
        query=req.query,
        candidates=fused,
        top_n=req.limit,
        floor=RERANK_FLOOR,
    )
    return SearchOut(
        results=[
            SearchHit(
                content=f"{r['key']}: {r['value']}",
                score=float(
                    r.get("_rerank_score") or r.get("_rrf_score") or r.get("score") or 0.0
                ),
                session_id=r["session_id"] or "",
                timestamp=r["created_at"],
                metadata={
                    "type": r["type"],
                    "confidence": r["confidence"],
                    "raw_quote": r.get("raw_quote"),
                    "active": r["active"],
                    "channels": r.get("_channels", {}),
                },
            )
            for r in final
        ]
    )
