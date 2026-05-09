"""Ingest pipeline: persist a turn and synchronously extract structured memories.

Per TASK.md §5 *"after POST /turns returns, the ingested data and extracted
memories must be immediately available via /recall"* — extraction is therefore
inline, not async. The eval harness gives us a 60s budget per turn.
"""

import logging

from memory import repository
from memory.schemas import TurnIn
from memory.services import extraction

log = logging.getLogger(__name__)


async def ingest_turn(turn: TurnIn) -> str:
    turn_id = await repository.insert_turn(turn)

    msgs = await repository.fetch_messages_for_turn(turn_id)
    inserted = 0
    try:
        inserted = await extraction.extract_and_store(
            turn_id=turn_id,
            user_id=turn.user_id,
            session_id=turn.session_id,
            messages=msgs,
        )
    except Exception as e:
        # Extraction is best-effort. Turn stays persisted; raw msgs remain
        # recallable via the Step 4 hybrid fallback (FTS over messages.content_tsv).
        log.exception("extraction_failed", extra={"turn_id": turn_id, "error": str(e)})

    log.info(
        "turn_ingested",
        extra={
            "turn_id": turn_id,
            "session_id": turn.session_id,
            "user_id": turn.user_id,
            "n_messages": len(msgs),
            "n_memories_inserted": inserted,
        },
    )
    return turn_id
