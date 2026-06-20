"""Typed data transforms used on task edges."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from llmasm.errors import ValidationError
from llmasm.schemas import ConversationText, RawText


TransformFn = Callable[[BaseModel, dict[str, Any] | None], BaseModel]


@dataclass(frozen=True)
class TransformSpec:
    """Registered transform metadata."""

    name: str
    from_schema: str
    to_schema: str
    fn: TransformFn


class TransformRegistry:
    """Registry for deterministic edge transforms."""

    def __init__(self) -> None:
        self._transforms: dict[str, TransformSpec] = {}

    def register(self, name: str, from_schema: str, to_schema: str, fn: TransformFn) -> None:
        """Register a transform."""

        self._transforms[name] = TransformSpec(name, from_schema, to_schema, fn)

    def resolve(self, name: str) -> TransformSpec:
        """Return a transform by name."""

        try:
            return self._transforms[name]
        except KeyError as exc:
            raise ValidationError(f"Unknown transform: {name}") from exc

    def can_transform(self, from_schema: str, to_schema: str, name: str | None) -> bool:
        """Return whether schemas are directly compatible or transformable."""

        if from_schema == to_schema:
            return True
        if name is None or name not in self._transforms:
            return False
        spec = self._transforms[name]
        return (spec.from_schema in {from_schema, "*"}) and spec.to_schema == to_schema

    def apply(
        self,
        name: str,
        value: BaseModel,
        metadata: dict[str, Any] | None = None,
    ) -> BaseModel:
        """Apply a registered transform."""

        return self.resolve(name).fn(value, metadata)


def _extract_text(value: BaseModel, _: dict[str, Any] | None = None) -> ConversationText:
    text = getattr(value, "text", None)
    if text is None:
        raise ValidationError("extract_text requires a text field")
    return ConversationText(text=str(text))


def _to_json_string(value: BaseModel, _: dict[str, Any] | None = None) -> RawText:
    return RawText(text=json.dumps(value.model_dump(mode="json"), sort_keys=True))


def _select_field(value: BaseModel, metadata: dict[str, Any] | None = None) -> RawText:
    field = (metadata or {}).get("field")
    if not field:
        raise ValidationError("select_field requires metadata.field")
    data = value.model_dump()
    if field not in data:
        raise ValidationError(f"select_field unknown field: {field}")
    return RawText(text=str(data[field]))


def default_transform_registry() -> TransformRegistry:
    """Return built-in transforms."""

    registry = TransformRegistry()
    registry.register("extract_text", "ConversationRecord", "ConversationText", _extract_text)
    registry.register("to_json_string", "*", "RawText", _to_json_string)
    registry.register("select_field", "*", "RawText", _select_field)
    return registry
