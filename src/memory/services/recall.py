"""Recall pipeline.

Step 4 (this version):
  1. Run BM25 over memories.value_tsv (top 30) and vector over memories.embedding
     (top 30) **in parallel**.
  2. RRF-fuse with k=60 → top 20.
  3. If memory channel is empty (extractor missed everything), fall back to
     BM25 over messages.content_tsv so the conversation text is still recallable.
  4. Format as bucketed prose.

Step 5 will insert Alem reranker between (2) and (4).
Step 8 replaces (4) with token-budget-aware priority assembly.
"""

import asyncio
import logging

from memory import repository
from memory.clients import embeddings
from memory.schemas import Citation, RecallIn, RecallOut, SearchHit, SearchIn, SearchOut
from memory.util.rrf import reciprocal_rank_fusion

log = logging.getLogger(__name__)

# Per-channel candidate breadth before fusion.
RETRIEVAL_K = 30
# How many fused results we keep for prose assembly.
FUSED_K = 20
DEFAULT_SNIPPET_CHARS = 280


async def recall(req: RecallIn) -> RecallOut:
    fused = await _hybrid_memories(
        query=req.query,
        user_id=req.user_id,
        session_id=req.session_id if req.user_id is None else None,
        per_channel=RETRIEVAL_K,
        fused_limit=FUSED_K,
    )

    if not fused:
        # Cold extraction fallback — search the raw conversation text.
        fallback = await repository.search_messages_by_bm25(
            req.query,
            user_id=req.user_id,
            session_id=req.session_id if req.user_id is None else None,
            limit=8,
        )
        if not fallback:
            return RecallOut(context="", citations=[])
        return _format_message_fallback(fallback)

    return _format_recall(fused)


async def _hybrid_memories(
    *,
    query: str,
    user_id: str | None,
    session_id: str | None,
    per_channel: int,
    fused_limit: int,
) -> list[dict]:
    qvec_task = asyncio.create_task(embeddings.embed(query))
    bm25_task = asyncio.create_task(
        repository.search_memories_by_bm25(
            query, user_id=user_id, session_id=session_id, limit=per_channel
        )
    )
    qvec = await qvec_task
    qlit = embeddings.to_pgvector(qvec)
    vec_rows = await repository.search_memories_by_embedding(
        qlit, user_id=user_id, session_id=session_id, limit=per_channel
    )
    bm25_rows = await bm25_task

    fused = reciprocal_rank_fusion(
        {"vector": vec_rows, "bm25": bm25_rows},
        id_key="id",
        k=60,
        limit=fused_limit,
    )
    return fused


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
    """When no extracted memory matched, surface raw conversation text. This
    is rare in practice but keeps a query from returning empty just because
    extraction missed something subtle."""
    lines = ["## Relevant from recent conversations"]
    citations: list[Citation] = []
    for r in rows:
        ts = r["timestamp"].strftime("%Y-%m-%d") if r.get("timestamp") else ""
        snippet = (r["content"] or "")[:DEFAULT_SNIPPET_CHARS]
        prefix = f"- [{ts}] " if ts else "- "
        lines.append(f"{prefix}{snippet}")
        citations.append(
            Citation(turn_id=r["turn_id"], score=float(r["score"]), snippet=snippet)
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
    score = float(r.get("_rrf_score") or r.get("score") or 0.0)
    return Citation(
        turn_id=r.get("source_turn") or "",
        score=score,
        snippet=snippet,
    )


# ── /search ───────────────────────────────────────────────────────────────


async def search(req: SearchIn) -> SearchOut:
    """Structured search. Hybrid memory retrieval — same pipeline as /recall
    but returns structured results instead of prose."""
    fused = await _hybrid_memories(
        query=req.query,
        user_id=req.user_id,
        session_id=req.session_id,
        per_channel=max(req.limit * 2, RETRIEVAL_K),
        fused_limit=req.limit,
    )
    return SearchOut(
        results=[
            SearchHit(
                content=f"{r['key']}: {r['value']}",
                score=float(r.get("_rrf_score") or r.get("score") or 0.0),
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
            for r in fused
        ]
    )
