import logging

from fastapi import APIRouter, Depends, status

from memory.auth import require_auth
from memory.schemas import TurnIn, TurnOut
from memory.services import ingest

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/turns", status_code=status.HTTP_201_CREATED, response_model=TurnOut)
async def post_turn(turn: TurnIn, _=Depends(require_auth)) -> TurnOut:
    turn_id = await ingest.ingest_turn(turn)
    return TurnOut(id=turn_id)
