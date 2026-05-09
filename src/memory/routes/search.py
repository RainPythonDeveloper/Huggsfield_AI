"""POST /search — stub for Step 2; real ranking lands in Steps 4-5."""

from fastapi import APIRouter, Depends

from memory.auth import require_auth
from memory.schemas import SearchIn, SearchOut

router = APIRouter()


@router.post("/search", response_model=SearchOut)
async def post_search(req: SearchIn, _=Depends(require_auth)) -> SearchOut:
    return SearchOut(results=[])
