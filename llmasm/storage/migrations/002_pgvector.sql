-- Migration 002: pgvector extension.
-- run_migrations() skips this file when the vector extension is unavailable
-- (checked via pg_available_extensions before applying).
-- The vector column DDL (ALTER TABLE + index) is applied at PostgresEmbeddingStore
-- initialisation time so the correct dimension count is available from RuntimeConfig.

CREATE EXTENSION IF NOT EXISTS vector;
