"""Apply SQL migrations idempotently on app startup.

This complements Postgres `docker-entrypoint-initdb.d` which only runs on a
fresh data dir. By reapplying migrations on every boot (each idempotent), we
support schema evolution without dropping the volume.
"""

import logging
from pathlib import Path

from memory.db import acquire

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def apply_migrations() -> None:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        log.warning("no_migrations_found", extra={"dir": str(MIGRATIONS_DIR)})
        return
    async with acquire() as conn:
        for f in files:
            sql = f.read_text(encoding="utf-8")
            try:
                await conn.execute(sql)
                log.info("migration_applied", extra={"file": f.name})
            except Exception as e:
                log.exception("migration_failed", extra={"file": f.name, "error": str(e)})
                raise
