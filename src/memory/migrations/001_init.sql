-- Memory service schema. See PLAN.md §3 for design rationale.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ── Conversation history ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS turns (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  TEXT NOT NULL,
    user_id     TEXT,
    timestamp   TIMESTAMPTZ NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw         JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS turns_session_idx ON turns(session_id);
CREATE INDEX IF NOT EXISTS turns_user_idx    ON turns(user_id);

CREATE TABLE IF NOT EXISTS messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id     UUID NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    name        TEXT,
    content     TEXT NOT NULL,
    position    INT NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS
                (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS messages_turn_idx ON messages(turn_id);
CREATE INDEX IF NOT EXISTS messages_tsv_idx  ON messages USING GIN(content_tsv);

-- ── Extracted memories ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT,
    session_id      TEXT,
    type            TEXT NOT NULL,
    key             TEXT NOT NULL,
    value           TEXT NOT NULL,
    raw_quote       TEXT,
    confidence      REAL NOT NULL DEFAULT 0.8,
    embedding       vector(1024) NOT NULL,
    value_tsv       tsvector GENERATED ALWAYS AS
                    (to_tsvector('english', key || ' ' || value)) STORED,
    source_turn     UUID REFERENCES turns(id) ON DELETE SET NULL,
    source_session  TEXT,
    supersedes      UUID REFERENCES memories(id) ON DELETE SET NULL,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS memories_user_active_idx ON memories(user_id) WHERE active;
CREATE INDEX IF NOT EXISTS memories_session_idx     ON memories(session_id);
CREATE INDEX IF NOT EXISTS memories_key_idx         ON memories(user_id, key) WHERE active;
CREATE INDEX IF NOT EXISTS memories_tsv_idx         ON memories USING GIN(value_tsv);
CREATE INDEX IF NOT EXISTS memories_embedding_idx   ON memories
    USING hnsw (embedding vector_cosine_ops);
