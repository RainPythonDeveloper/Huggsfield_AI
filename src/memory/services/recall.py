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

    if not final_rows:
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
        return _format_message_fallback(fallback)

    return _format_recall(final_rows)


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
    """
    bm25_task = asyncio.create_task(
        repository.search_memories_by_bm25(
            query, user_id=user_id, session_id=session_id, limit=per_channel
        )
    )

    vec_rows: list[dict] = []
    try:
        qvec = await embeddings.embed(query)
        qlit = embeddings.to_pgvector(qvec)
        vec_rows = await repository.search_memories_by_embedding(
            qlit, user_id=user_id, session_id=session_id, limit=per_channel
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


# ── Formatters ─────────────────────────────────────────────────────────────


def _format_recall(rows: list[dict]) -> RecallOut:
    facts = [r for r in rows if r["type"] in ("fact", "preference", "relation")]
    other = [r for r in rows if r["type"] not in ("fact", "preference", "relation")]

    lines: list[str] = []
    citations: list[Citation] = []

    if facts:
        lines.append("## Known facts about this user")
        for r in facts:
            lines.append(f"- {_humanize(r)}")
            citations.append(_cite(r))
    if other:
        if lines:
            lines.append("")
        lines.append("## Relevant from recent conversations")
        for r in other:
            lines.append(f"- {_humanize(r)}")
            citations.append(_cite(r))
    return RecallOut(context="\n".join(lines).strip(), citations=citations)


def _format_message_fallback(rows: list[dict]) -> RecallOut:
    lines = ["## Relevant from recent conversations"]
    citations: list[Citation] = []
    for r in rows:
        ts = r["timestamp"].strftime("%Y-%m-%d") if r.get("timestamp") else ""
        snippet = (r["content"] or "")[:DEFAULT_SNIPPET_CHARS]
        prefix = f"- [{ts}] " if ts else "- "
        lines.append(f"{prefix}{snippet}")
        score = float(r.get("_rerank_score") or r.get("score") or 0.0)
        citations.append(
            Citation(turn_id=r["turn_id"], score=score, snippet=snippet)
        )
    return RecallOut(context="\n".join(lines), citations=citations)


def _humanize(r: dict) -> str:
    key = r["key"].replace("_", " ")
    value = r["value"]
    quote = r.get("raw_quote")
    if quote and len(quote) < 140:
        return f"{key}: {value} (\"{quote}\")"
    return f"{key}: {value}"


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
