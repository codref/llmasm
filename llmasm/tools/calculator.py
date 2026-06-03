"""Safe calculator tool that evaluates mathematical expressions."""

from __future__ import annotations

import math

from pydantic import BaseModel

from llmasm.schemas import RawText
from llmasm.tools.base import ToolSpec

_SAFE_BUILTINS = {
    "abs": abs,
    "float": float,
    "int": int,
    "max": max,
    "min": min,
    "round": round,
    "sum": sum,
    "pow": pow,
    "True": True,
    "False": False,
}

_SAFE_GLOBALS: dict[str, object] = {
    "__builtins__": _SAFE_BUILTINS,
    **{name: getattr(math, name) for name in dir(math) if not name.startswith("_")},
}


class CalculatorTool:
    """Evaluate a mathematical expression using a restricted Python eval."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator.eval",
            description="Evaluate a mathematical expression. Supports arithmetic, trigonometry, logarithms, exponents, and constants like pi and e.",
            input_schema="RawText",
            output_schema="RawText",
        )

    def invoke(self, input: BaseModel) -> BaseModel:
        expression = getattr(input, "text", str(input)).strip()
        if not expression:
            return RawText(text="No expression provided.")
        try:
            result = eval(expression, _SAFE_GLOBALS, {})
            return RawText(text=str(result))
        except Exception as exc:
            return RawText(text=f"Error: {exc}")
