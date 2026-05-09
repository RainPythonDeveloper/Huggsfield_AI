-- Step 2: temporary embedding column on messages for naive recall baseline.
-- This column will be removed in Step 3 once memories.embedding becomes the
-- primary recall surface; we keep messages.content_tsv as the cold-extraction
-- fallback.

ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS embedding vector(1024);

-- HNSW index over message embeddings. NULLs are skipped automatically by the
-- distance operator at query time; no partial clause needed.
CREATE INDEX IF NOT EXISTS messages_embedding_idx
    ON messages USING hnsw (embedding vector_cosine_ops);
