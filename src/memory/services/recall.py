"""Step 3: recall over extracted memories.

Pipeline:
  embed(query) → cosine top-k over `memories.embedding` (active only)
                → format as readable prose with citations.

Hybrid retrieval (BM25 + raw-message fallback) lands in Step 4.
Reranking lands in Step 5. Budget-aware assembly lands in Step 8.
"""

import logging

from memory import repository
from memory.clients import embeddings
from memory.schemas import Citation, RecallIn, RecallOut, SearchHit, SearchIn, SearchOut

log = logging.getLogger(__name__)

DEFAULT_RECALL_K = 8
DEFAULT_SNIPPET_CHARS = 280


async def recall(req: RecallIn) -> RecallOut:
    qvec = await embeddings.embed(req.query)
    qlit = embeddings.to_pgvector(qvec)
    rows = await repository.search_memories_by_embedding(
        qlit,
        user_id=req.user_id,
        session_id=req.session_id if req.user_id is None else None,
        limit=DEFAULT_RECALL_K,
    )
    if not rows:
        return RecallOut(context="", citations=[])

    return _format_recall(rows)


def _format_recall(rows: list[dict]) -> RecallOut:
    """Bucket extracted memories into readable prose. Step 8 replaces this with
    priority-aware bucketing under a token budget."""
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
        lines.append("")
        lines.append("## Relevant from recent conversations")
        for r in other:
            lines.append(f"- {_humanize(r)}")
            citations.append(_cite(r))
    return RecallOut(context="\n".join(lines).strip(), citations=citations)


def _humanize(r: dict) -> str:
    """Render a memory row as a human-readable bullet."""
    key = r["key"].replace("_", " ")
    value = r["value"]
    quote = r.get("raw_quote")
    if quote and len(quote) < 140:
        return f"{key}: {value} (\"{quote}\")"
    return f"{key}: {value}"


def _cite(r: dict) -> Citation:
    snippet = (r.get("raw_quote") or f"{r['key']}: {r['value']}")[:DEFAULT_SNIPPET_CHARS]
    return Citation(
        turn_id=r.get("source_turn") or "",
        score=float(r["score"]),
        snippet=snippet,
    )


# ── /search ───────────────────────────────────────────────────────────────
async def search(req: SearchIn) -> SearchOut:
    """Structured search over memories. Returns key/value as content."""
    qvec = await embeddings.embed(req.query)
    qlit = embeddings.to_pgvector(qvec)
    rows = await repository.search_memories_by_embedding(
        qlit,
        user_id=req.user_id,
        session_id=req.session_id,
        limit=req.limit,
    )
    return SearchOut(
        results=[
            SearchHit(
                content=f"{r['key']}: {r['value']}",
                score=float(r["score"]),
                session_id=r["session_id"] or "",
                timestamp=r["created_at"],
                metadata={
                    "type": r["type"],
                    "confidence": r["confidence"],
                    "raw_quote": r.get("raw_quote"),
                    "active": r["active"],
                },
            )
            for r in rows
        ]
    )
