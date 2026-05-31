"""Built-in schema models used by task graph ports."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ConversationRecord(BaseModel):
    """A retrieved conversation."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationText(BaseModel):
    """Plain conversation text extracted from a record."""

    text: str


class Summary(BaseModel):
    """Summary text with optional source provenance."""

    text: str
    source_id: str | None = None


class WeatherQuery(BaseModel):
    """Historical weather lookup input."""

    location: str | None = None
    date: str | None = None


class WeatherObservation(BaseModel):
    """Weather lookup output."""

    condition: str
    source_url: str | None = None


class FinalAnswer(BaseModel):
    """User-facing final answer."""

    text: str
    sources: list[str] = Field(default_factory=list)


class RawText(BaseModel):
    """Arbitrary text."""

    text: str


class JsonValue(BaseModel):
    """JSON-compatible value wrapper."""

    value: dict[str, Any] | list[Any] | str | int | float | bool | None


class NotFound(BaseModel):
    """Tool result for a missing requested resource."""

    resource_type: str
    resource_id: str
    detail: str


class RoutingDecision(BaseModel):
    """Router node output selecting a downstream branch."""

    selected_branch: str
    reason: str | None = None
