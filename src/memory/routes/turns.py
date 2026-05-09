import logging

from fastapi import APIRouter, Depends, status

from memory import repository
from memory.auth import require_auth
from memory.schemas import TurnIn, TurnOut

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/turns", status_code=status.HTTP_201_CREATED, response_model=TurnOut)
async def post_turn(turn: TurnIn, _=Depends(require_auth)) -> TurnOut:
    turn_id = await repository.insert_turn(turn)
    log.info(
        "turn_persisted",
        extra={
            "turn_id": turn_id,
            "session_id": turn.session_id,
            "user_id": turn.user_id,
            "n_messages": len(turn.messages),
        },
    )
    # Step 3 will hook extraction here. For now: raw store only.
    return TurnOut(id=turn_id)
