#!/usr/bin/env python3
"""Reset pgvector embeddings so a new embedding model can be used.

Drops the vector column + index and clears the embeddings table.
Next run will auto-create the column with the correct dimension.

Usage:
    python -m llmasm.tools.reset_embeddings postgresql://llmasm:llmasm@localhost:15432/llmasm
"""

from __future__ import annotations

import argparse
import sys

import psycopg


def reset_embeddings(dsn: str) -> None:
    conn = psycopg.connect(dsn, autocommit=True)
    with conn.cursor() as cur:
        # Drop index first (depends on column)
        cur.execute("DROP INDEX IF EXISTS idx_embeddings_vector")
        # Drop pgvector column
        cur.execute("ALTER TABLE embeddings DROP COLUMN IF EXISTS vector")
        # Clear all embeddings so they are re-generated on next run
        cur.execute("DELETE FROM embeddings")
    print("Embeddings reset: vector column dropped, table cleared.")
    print("Next run will auto-create the vector column with the correct dimension.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset pgvector embeddings for a new model")
    parser.add_argument("dsn", help="PostgreSQL DSN")
    args = parser.parse_args()
    reset_embeddings(args.dsn)


if __name__ == "__main__":
    main()
