"""Optional bearer-token auth dependency. If MEMORY_AUTH_TOKEN is unset, auth is bypassed."""

from fastapi import Header, HTTPException, status

from memory.config import get_settings


async def require_auth(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.memory_auth_token:
        return  # auth disabled
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.memory_auth_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")
