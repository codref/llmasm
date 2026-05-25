"""Tool registry."""

from __future__ import annotations

from llmasm.errors import ValidationError
from llmasm.graph.registry import SchemaRegistry
from llmasm.tools.base import Tool


class ToolRegistry:
    """Registry of executable tools."""

    def __init__(self, schema_registry: SchemaRegistry) -> None:
        self._schema_registry = schema_registry
        self._tools: dict[str, Tool] = {}

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
