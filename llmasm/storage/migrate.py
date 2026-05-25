"""Explicit migration runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_migrations(conn: Any) -> int:
    """Apply unapplied SQL migrations in order and return the count."""

    migrations = sorted((Path(__file__).parent / "migrations").glob("*.sql"))
    with conn.cursor() as cur:
        cur.execute(
            "create table if not exists schema_version "
            "(version integer primary key, applied_at timestamptz not null default now())"
        )
        cur.execute("select version from schema_version")
        applied = {row[0] for row in cur.fetchall()}
    count = 0
    for migration in migrations:
        version = int(migration.name.split("_", 1)[0])
        if version in applied:
            continue
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(migration.read_text())
                cur.execute("insert into schema_version(version) values (%s)", (version,))
        count += 1
    return count
