"""Schema registry."""

from __future__ import annotations

from pydantic import BaseModel

from llmasm.errors import ValidationError
from llmasm import schemas


class SchemaRegistry:
    """Registry mapping schema tags to Pydantic models."""

    def __init__(self) -> None:
        self._models: dict[str, type[BaseModel]] = {}

    def register(self, tag: str, model: type[BaseModel]) -> None:
        """Register a schema tag."""

        self._models[tag] = model

    def get(self, tag: str) -> type[BaseModel]:
        """Return the model for a tag."""

        try:
            return self._models[tag]
        except KeyError as exc:
            raise ValidationError(f"Unknown schema tag: {tag}") from exc

    def has(self, tag: str) -> bool:
        """Return whether the tag is known."""

        return tag in self._models

    def describe(self) -> str:
        """Render deterministic prompt descriptions."""

        lines = []
        for tag in sorted(self._models):
            model = self._models[tag]
            fields = ", ".join(model.model_fields)
            lines.append(f"- {tag}({fields})")
        return "\n".join(lines)


def default_schema_registry() -> SchemaRegistry:
    """Return a registry with built-in schemas."""

    registry = SchemaRegistry()
    for model in (
        schemas.ConversationRecord,
        schemas.ConversationText,
        schemas.Summary,
        schemas.WeatherQuery,
        schemas.WeatherObservation,
        schemas.FinalAnswer,
        schemas.RawText,
        schemas.JsonValue,
        schemas.NotFound,
        schemas.RoutingDecision,
        schemas.ToolCallRequest,
        schemas.ToolCallResult,
        schemas.ToolDefinition,
    ):
        registry.register(model.__name__, model)
    return registry
