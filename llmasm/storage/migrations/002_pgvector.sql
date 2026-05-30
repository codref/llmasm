-- Migration 002: pgvector extension and native vector column.
-- run_migrations() skips this file when the vector extension is unavailable
-- (checked via pg_available_extensions before applying).

CREATE EXTENSION IF NOT EXISTS vector;

-- 768 dimensions matches nomic-embed-text (default embedding_model in RuntimeConfig).
-- Adjust if a different model with different output dimensions is used.
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS vector vector(768);

CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON embeddings USING ivfflat (vector vector_cosine_ops)
    WHERE vector IS NOT NULL;
