import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

from memory.config import get_settings

log = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool
    settings = get_settings()
    _pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=1,
        max_size=10,
        command_timeout=30,
    )
    log.info("db_pool_initialized")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        log.info("db_pool_closed")


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialized")
    return _pool


@asynccontextmanager
async def acquire() -> AsyncIterator[asyncpg.Connection]:
    pool = get_pool()
    async with pool.acquire() as conn:
        yield conn


async def ping() -> bool:
    try:
        async with acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        log.warning("db_ping_failed", extra={"error": str(e)})
        return False
