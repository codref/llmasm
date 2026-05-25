"""Postgres storage placeholder.

The in-memory storage contract is the runnable v0 backend. This module exists so
imports are stable while the full SQL adapter is implemented behind the same
Storage protocol.
"""

from __future__ import annotations

from llmasm.errors import StorageError


class PostgresStorage:
    """Placeholder for the plain Postgres backend."""

    def __init__(self, conn: object) -> None:
        self.conn = conn
        raise StorageError("PostgresStorage is not implemented in this in-memory v0 slice")
