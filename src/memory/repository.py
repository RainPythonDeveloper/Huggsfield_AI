"""Thin async repository over asyncpg. No business logic — just CRUD."""

import json
import logging
from typing import Any

from memory.db import acquire
from memory.schemas import Message, TurnIn

log = logging.getLogger(__name__)


# ── Turns / messages ───────────────────────────────────────────────────────


async def insert_turn(turn: TurnIn) -> str:
    """Persist a turn and its messages atomically. Returns the new turn UUID."""
    async with acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO turns (session_id, user_id, timestamp, metadata, raw)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                RETURNING id::text
                """,
                turn.session_id,
                turn.user_id,
                turn.timestamp,
                json.dumps(turn.metadata),
                turn.model_dump_json(),
            )
            turn_id = row["id"]
            await _insert_messages(conn, turn_id, turn.messages)
    return turn_id


async def _insert_messages(conn, turn_id: str, messages: list[Message]) -> None:
    if not messages:
        return
    await conn.executemany(
        """
        INSERT INTO messages (turn_id, role, name, content, position)
        VALUES ($1::uuid, $2, $3, $4, $5)
        """,
        [(turn_id, m.role, m.name, m.content, i) for i, m in enumerate(messages)],
    )


async def fetch_turn_messages(turn_id: str) -> list[dict[str, Any]]:
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, name, content, position
            FROM messages
            WHERE turn_id = $1::uuid
            ORDER BY position ASC
            """,
            turn_id,
        )
    return [dict(r) for r in rows]


# ── Cleanup ────────────────────────────────────────────────────────────────


async def delete_session(session_id: str) -> None:
    """Delete all data for a session: turns (cascades messages), memories."""
    async with acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM memories WHERE session_id = $1", session_id)
            await conn.execute("DELETE FROM turns WHERE session_id = $1", session_id)


async def delete_user(user_id: str) -> None:
    """Delete all data for a user across all sessions."""
    async with acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM memories WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM turns WHERE user_id = $1", user_id)


# ── Memories ───────────────────────────────────────────────────────────────


async def list_user_memories(user_id: str) -> list[dict[str, Any]]:
    """Return ALL memories (active + superseded) for a user, newest-first."""
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, type, key, value, confidence,
                   source_session, source_turn::text AS source_turn,
                   created_at, updated_at,
                   supersedes::text AS supersedes,
                   active
            FROM memories
            WHERE user_id = $1
            ORDER BY created_at DESC
            """,
            user_id,
        )
    return [dict(r) for r in rows]
