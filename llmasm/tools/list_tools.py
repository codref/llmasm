"""Built-in tool for listing registered tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from llmasm.schemas import JsonValue
from llmasm.tools.base import ToolSpec

if TYPE_CHECKING:
    from llmasm.tools.registry import ToolRegistry


class ToolsListTool:
    """Return the names and descriptions of every registered tool."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="tools.list",
            description="List all registered tools with their names and descriptions.",
            input_schema="RawText",
            output_schema="JsonValue",
        )

    def invoke(self, input: BaseModel) -> BaseModel:
        text = getattr(input, "text", "")
        _ = text  # input is ignored; kept for protocol compatibility
        tools = []
        for name in self._registry.names():
            spec = self._registry.get(name).spec()
            tools.append({"name": spec.name, "description": spec.description})
        return JsonValue(value=tools)
