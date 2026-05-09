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


async def fetch_messages_for_turn(turn_id: str) -> list[dict[str, Any]]:
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, role, name, content, position
            FROM messages
            WHERE turn_id = $1::uuid
            ORDER BY position ASC
            """,
            turn_id,
        )
    return [dict(r) for r in rows]


async def update_message_embedding(message_id: str, embedding_pgliteral: str) -> None:
    async with acquire() as conn:
        await conn.execute(
            "UPDATE messages SET embedding = $1::vector WHERE id = $2::uuid",
            embedding_pgliteral,
            message_id,
        )


async def search_messages_by_embedding(
    embedding_pgliteral: str,
    *,
    user_id: str | None,
    session_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Naive cosine top-k over message embeddings.

    Filtering rules:
      - if user_id given: only that user's turns; no session filter (cross-session is intentional)
      - if user_id is None but session_id given: scope to that session
      - if both None: global (mostly used for /search)
    """
    where_parts = ["m.embedding IS NOT NULL"]
    args: list[Any] = [embedding_pgliteral]
    if user_id is not None:
        args.append(user_id)
        where_parts.append(f"t.user_id = ${len(args)}")
    elif session_id is not None:
        args.append(session_id)
        where_parts.append(f"t.session_id = ${len(args)}")
    args.append(limit)
    where_sql = " AND ".join(where_parts)
    sql = f"""
        SELECT m.id::text         AS message_id,
               m.turn_id::text    AS turn_id,
               m.content,
               t.session_id,
               t.user_id,
               t.timestamp,
               1 - (m.embedding <=> $1::vector) AS score
        FROM messages m
        JOIN turns t ON t.id = m.turn_id
        WHERE {where_sql}
        ORDER BY m.embedding <=> $1::vector ASC
        LIMIT ${len(args)}
    """
    async with acquire() as conn:
        rows = await conn.fetch(sql, *args)
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


async def insert_memory(
    *,
    user_id: str,
    session_id: str | None,
    type_: str,
    key: str,
    value: str,
    confidence: float,
    raw_quote: str | None,
    source_turn: str | None,
    embedding_pgliteral: str,
) -> str:
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO memories (
                user_id, session_id, type, key, value, confidence,
                raw_quote, source_turn, source_session, embedding, active
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::uuid, $2, $9::vector, TRUE)
            RETURNING id::text
            """,
            user_id,
            session_id,
            type_,
            key,
            value,
            confidence,
            raw_quote,
            source_turn,
            embedding_pgliteral,
        )
    return row["id"]


async def search_memories_by_embedding(
    embedding_pgliteral: str,
    *,
    user_id: str | None,
    session_id: str | None,
    limit: int,
    only_active: bool = True,
) -> list[dict[str, Any]]:
    """Cosine top-k over memory embeddings.

    Scoping: by default we filter to user_id (cross-session is intentional —
    a user's facts are theirs across all sessions). Only when user_id is None
    do we fall back to session-scoping or global search.
    """
    where_parts: list[str] = []
    args: list[Any] = [embedding_pgliteral]
    if user_id is not None:
        args.append(user_id)
        where_parts.append(f"user_id = ${len(args)}")
    elif session_id is not None:
        args.append(session_id)
        where_parts.append(f"session_id = ${len(args)}")
    if only_active:
        where_parts.append("active = TRUE")
    args.append(limit)
    where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    sql = f"""
        SELECT id::text,
               user_id, session_id, type, key, value, confidence,
               raw_quote, source_turn::text AS source_turn,
               created_at, updated_at, active,
               1 - (embedding <=> $1::vector) AS score
        FROM memories
        {where_sql}
        ORDER BY embedding <=> $1::vector ASC
        LIMIT ${len(args)}
    """
    async with acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


async def search_memories_by_bm25(
    query: str,
    *,
    user_id: str | None,
    session_id: str | None,
    limit: int,
    only_active: bool = True,
) -> list[dict[str, Any]]:
    """BM25-style FTS over `memories.value_tsv` (key + value tokens).

    Uses `plainto_tsquery` so user query terms are tokenised the same way as
    the indexed content. `ts_rank_cd` is cover-density ranking — favours hits
    where query terms appear close together.
    """
    where_parts: list[str] = ["m.value_tsv @@ plainto_tsquery('english', $1)"]
    args: list[Any] = [query]
    if user_id is not None:
        args.append(user_id)
        where_parts.append(f"m.user_id = ${len(args)}")
    elif session_id is not None:
        args.append(session_id)
        where_parts.append(f"m.session_id = ${len(args)}")
    if only_active:
        where_parts.append("m.active = TRUE")
    args.append(limit)
    where_sql = " AND ".join(where_parts)
    sql = f"""
        SELECT m.id::text,
               m.user_id, m.session_id, m.type, m.key, m.value, m.confidence,
               m.raw_quote, m.source_turn::text AS source_turn,
               m.created_at, m.updated_at, m.active,
               ts_rank_cd(m.value_tsv, plainto_tsquery('english', $1)) AS score
        FROM memories m
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT ${len(args)}
    """
    async with acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


async def search_messages_by_bm25(
    query: str,
    *,
    user_id: str | None,
    session_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """FTS fallback over raw `messages.content_tsv`. Used when the extractor
    missed something — the conversation text itself is still queryable.
    """
    where_parts: list[str] = ["m.content_tsv @@ plainto_tsquery('english', $1)"]
    args: list[Any] = [query]
    if user_id is not None:
        args.append(user_id)
        where_parts.append(f"t.user_id = ${len(args)}")
    elif session_id is not None:
        args.append(session_id)
        where_parts.append(f"t.session_id = ${len(args)}")
    args.append(limit)
    where_sql = " AND ".join(where_parts)
    # Skip assistant utterances — extraction targets user-stated facts.
    where_sql += " AND m.role IN ('user', 'tool')"
    sql = f"""
        SELECT m.id::text             AS message_id,
               m.turn_id::text        AS turn_id,
               m.role, m.content,
               t.session_id, t.user_id, t.timestamp,
               ts_rank_cd(m.content_tsv, plainto_tsquery('english', $1)) AS score
        FROM messages m
        JOIN turns t ON t.id = m.turn_id
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT ${len(args)}
    """
    async with acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


async def fetch_recent_messages_for_session(
    session_id: str, limit: int = 6
) -> list[dict[str, Any]]:
    """Recent raw messages — used in /recall as a low-priority fallback bucket."""
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT m.id::text AS message_id, m.turn_id::text, m.role, m.content,
                   t.timestamp, t.session_id
            FROM messages m
            JOIN turns t ON t.id = m.turn_id
            WHERE t.session_id = $1
            ORDER BY t.timestamp DESC, m.position DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )
    return [dict(r) for r in rows]
