"""POST /recall — stub for Step 2; real hybrid retrieval lands in Steps 4-8."""

from fastapi import APIRouter, Depends

from memory.auth import require_auth
from memory.schemas import RecallIn, RecallOut

router = APIRouter()


@router.post("/recall", response_model=RecallOut)
async def post_recall(req: RecallIn, _=Depends(require_auth)) -> RecallOut:
    # Step 2 fills this in with naive embedding top-k.
    return RecallOut(context="", citations=[])
