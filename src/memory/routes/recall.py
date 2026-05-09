from fastapi import APIRouter, Depends

from memory.auth import require_auth
from memory.schemas import RecallIn, RecallOut
from memory.services import recall as recall_svc

router = APIRouter()


@router.post("/recall", response_model=RecallOut)
async def post_recall(req: RecallIn, _=Depends(require_auth)) -> RecallOut:
    return await recall_svc.recall(req)
