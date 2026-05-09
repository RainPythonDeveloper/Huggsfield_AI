from fastapi import APIRouter, Depends

from memory import repository
from memory.auth import require_auth
from memory.schemas import MemoriesOut, MemoryOut

router = APIRouter()


@router.get("/users/{user_id}/memories", response_model=MemoriesOut)
async def get_user_memories(user_id: str, _=Depends(require_auth)) -> MemoriesOut:
    rows = await repository.list_user_memories(user_id)
    return MemoriesOut(memories=[MemoryOut(**r) for r in rows])
