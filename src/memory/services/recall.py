"""Step 2: naive embedding-only recall.

Hybrid retrieval, reranking, query rewriting, and budget-aware assembly land
in subsequent steps. We start with vanilla cosine top-k to establish a
recall@5 baseline against the fixture.
"""

import logging

from memory import repository
from memory.clients import embeddings
from memory.schemas import Citation, RecallIn, RecallOut, SearchHit, SearchIn, SearchOut

log = logging.getLogger(__name__)

DEFAULT_RECALL_K = 5
DEFAULT_SNIPPET_CHARS = 280


async def recall(req: RecallIn) -> RecallOut:
    qvec = await embeddings.embed(req.query)
    qlit = embeddings.to_pgvector(qvec)
    rows = await repository.search_messages_by_embedding(
        qlit,
        user_id=req.user_id,
        session_id=req.session_id if req.user_id is None else None,
        limit=DEFAULT_RECALL_K,
    )
    if not rows:
        return RecallOut(context="", citations=[])

    # Naive prose: list top hits as bullets. Step 8 replaces this with
    # priority-aware bucketing under a token budget.
    lines = ["## Relevant from recent conversations"]
    citations: list[Citation] = []
    for r in rows:
        ts = r["timestamp"].strftime("%Y-%m-%d") if r["timestamp"] else ""
        snippet = (r["content"] or "")[:DEFAULT_SNIPPET_CHARS]
        prefix = f"- [{ts}] " if ts else "- "
        lines.append(f"{prefix}{snippet}")
        citations.append(
            Citation(turn_id=r["turn_id"], score=float(r["score"]), snippet=snippet)
        )
    return RecallOut(context="\n".join(lines), citations=citations)


async def search(req: SearchIn) -> SearchOut:
    qvec = await embeddings.embed(req.query)
    qlit = embeddings.to_pgvector(qvec)
    rows = await repository.search_messages_by_embedding(
        qlit,
        user_id=req.user_id,
        session_id=req.session_id,
        limit=req.limit,
    )
    return SearchOut(
        results=[
            SearchHit(
                content=r["content"],
                score=float(r["score"]),
                session_id=r["session_id"],
                timestamp=r["timestamp"],
                metadata={},
            )
            for r in rows
        ]
    )
