"""Tool interfaces."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class ToolSpec(BaseModel):
    """Static tool description."""

    name: str
    description: str
    input_schema: str
    output_schema: str


class Tool(Protocol):
    """Protocol for executable tools."""

    def spec(self) -> ToolSpec:
        """Return the tool specification."""

    def invoke(self, input: BaseModel, provider: Any = None) -> BaseModel:
        """Invoke the tool.  *provider* is the active LLM provider, if available."""
