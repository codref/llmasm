"""Tool interfaces."""

from __future__ import annotations

from typing import Callable, Protocol

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

    def invoke(self, input: BaseModel) -> BaseModel:
        """Invoke the tool."""


class _FunctionTool:
    """Adapter turning a Pydantic-typed function into a Tool."""

    def __init__(
        self,
        func: Callable[[BaseModel], BaseModel],
        input_schema: type[BaseModel],
        output_schema: str,
        name: str,
        description: str,
    ) -> None:
        self._func = func
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._name = name
        self._description = description

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=self._description,
            input_schema=self._input_schema.__name__,
            output_schema=self._output_schema,
        )

    def invoke(self, input: BaseModel) -> BaseModel:
        return self._func(input)


def make_tool(
    func: Callable[[BaseModel], BaseModel],
    input_schema: type[BaseModel],
    output_schema: str = "RawText",
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Build a Tool from a function and a Pydantic input model."""

    tool_name = name or func.__name__
    tool_description = description or func.__doc__ or tool_name
    return _FunctionTool(
        func=func,
        input_schema=input_schema,
        output_schema=output_schema,
        name=tool_name,
        description=tool_description,
    )


def function_tool(
    input_schema: type[BaseModel],
    output_schema: str = "RawText",
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[[BaseModel], BaseModel]], Tool]:
    """Decorator that builds a Tool from a Pydantic-typed function."""

    def decorator(func: Callable[[BaseModel], BaseModel]) -> Tool:
        return make_tool(
            func=func,
            input_schema=input_schema,
            output_schema=output_schema,
            name=name,
            description=description,
        )

    return decorator
