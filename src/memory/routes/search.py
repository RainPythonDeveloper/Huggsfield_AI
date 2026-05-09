from fastapi import APIRouter, Depends

from memory.auth import require_auth
from memory.schemas import SearchIn, SearchOut
from memory.services import recall as recall_svc

router = APIRouter()


@router.post("/search", response_model=SearchOut)
async def post_search(req: SearchIn, _=Depends(require_auth)) -> SearchOut:
    return await recall_svc.search(req)
