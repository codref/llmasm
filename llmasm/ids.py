"""ID helpers."""

from __future__ import annotations

from uuid import uuid4

from llmasm.errors import ValidationError

ACCEPTED_PREFIXES = {
    "workspace",
    "taskgraph",
    "run",
    "node",
    "edge",
    "artifact",
    "goal",
    "memory",
    "checkpoint",
}


def new_id(prefix: str) -> str:
    """Return a new prefixed identifier."""

    if prefix not in ACCEPTED_PREFIXES:
        raise ValidationError(f"Unsupported ID prefix: {prefix}")
    return f"{prefix}_{uuid4().hex}"
