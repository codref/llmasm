"""Tool registry."""

from __future__ import annotations

from typing import Any

from llmasm.errors import ValidationError
from llmasm.graph.registry import SchemaRegistry
from llmasm.tools.base import Tool


class ToolRegistry:
    """Registry of executable tools."""

    def __init__(self, schema_registry: SchemaRegistry) -> None:
        self._schema_registry = schema_registry
        self._tools: dict[str, Tool] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        """Register tools that ship with the registry."""

        from llmasm.tools.list_tools import ToolsListTool

        self.register(ToolsListTool(self))

    def register(self, tool: Tool) -> None:
        """Register a tool after validating schema tags."""

        spec = tool.spec()
        for tag in (spec.input_schema, spec.output_schema):
            if not self._schema_registry.has(tag):
                raise ValidationError(f"Tool {spec.name} references unknown schema {tag}")
        self._tools[spec.name] = tool

    def get(self, name: str) -> Tool:
        """Return a registered tool."""

        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValidationError(f"Unknown tool: {name}") from exc

    def has(self, name: str) -> bool:
        """Return whether a tool is registered."""

        return name in self._tools

    def describe(self) -> str:
        """Render deterministic prompt descriptions."""

        lines = []
        for name in sorted(self._tools):
            spec = self._tools[name].spec()
            lines.append(
                f"- {spec.name}: {spec.description} "
                f"({spec.input_schema} -> {spec.output_schema})"
            )
        return "\n".join(lines)

    def names(self) -> list[str]:
        """Return sorted registered tool names."""

        return sorted(self._tools)

    def to_json_schema(self, name: str) -> dict[str, Any]:
        """Return an LLM-compatible JSON Schema definition for a registered tool."""

        tool = self.get(name)
        spec = tool.spec()
        model = self._schema_registry.get(spec.input_schema)
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": model.model_json_schema(),
            },
        }
