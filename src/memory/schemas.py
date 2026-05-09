"""Pydantic schemas — wire contract per TASK.md §3."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ── /turns ─────────────────────────────────────────────────────────────────


class Message(BaseModel):
    role: Literal["user", "assistant", "tool", "system"]
    name: str | None = None
    content: str


class TurnIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=256)
    user_id: str | None = None
    messages: list[Message] = Field(min_length=1)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnOut(BaseModel):
    id: str


# ── /recall ────────────────────────────────────────────────────────────────


class RecallIn(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    user_id: str | None = None
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallOut(BaseModel):
    context: str
    citations: list[Citation]


# ── /search ────────────────────────────────────────────────────────────────


class SearchIn(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None
    user_id: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class SearchHit(BaseModel):
    content: str
    score: float
    session_id: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchOut(BaseModel):
    results: list[SearchHit]


# ── /users/{user_id}/memories ──────────────────────────────────────────────


class MemoryOut(BaseModel):
    id: str
    type: str
    key: str
    value: str
    confidence: float
    source_session: str | None
    source_turn: str | None
    created_at: datetime
    updated_at: datetime
    supersedes: str | None
    active: bool


class MemoriesOut(BaseModel):
    memories: list[MemoryOut]
