"""Step 2 ingest pipeline: persist turn → embed each message → backfill vectors.

In Step 3 this gets wrapped with LLM-extraction; for now it's the recall surface.
"""

import logging

from memory import repository
from memory.clients import embeddings
from memory.schemas import TurnIn

log = logging.getLogger(__name__)


async def ingest_turn(turn: TurnIn) -> str:
    """Persist a turn and embed its messages. Returns the new turn UUID."""
    turn_id = await repository.insert_turn(turn)

    # Best-effort embedding. If the embeddings backend is unavailable the
    # turn still persists (recallable later via tsvector fallback in Step 4),
    # but we surface the error to the caller as a 500 so they know.
    msgs = await repository.fetch_messages_for_turn(turn_id)
    contents = [m["content"] for m in msgs]
    vectors = await embeddings.embed_many(contents)
    for m, v in zip(msgs, vectors, strict=True):
        await repository.update_message_embedding(
            m["id"], embeddings.to_pgvector(v)
        )

    log.info(
        "turn_ingested",
        extra={
            "turn_id": turn_id,
            "session_id": turn.session_id,
            "user_id": turn.user_id,
            "n_messages_embedded": len(vectors),
        },
    )
    return turn_id
