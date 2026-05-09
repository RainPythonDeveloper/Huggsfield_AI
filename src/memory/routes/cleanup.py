import logging

from fastapi import APIRouter, Depends, Response, status

from memory import repository
from memory.auth import require_auth

log = logging.getLogger(__name__)
router = APIRouter()


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, _=Depends(require_auth)) -> Response:
    await repository.delete_session(session_id)
    log.info("session_deleted", extra={"session_id": session_id})
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str, _=Depends(require_auth)) -> Response:
    await repository.delete_user(user_id)
    log.info("user_deleted", extra={"user_id": user_id})
    return Response(status_code=status.HTTP_204_NO_CONTENT)
