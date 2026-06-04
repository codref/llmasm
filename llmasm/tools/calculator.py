"""Safe calculator tool that evaluates mathematical expressions."""

from __future__ import annotations

import math
from typing import Any

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
            description="Evaluate a mathematical expression. Accepts natural language (e.g. 'factorial of 10', 'square root of 16') and Python expressions.",
            input_schema="RawText",
            output_schema="RawText",
        )

    def invoke(self, input: BaseModel, provider: Any = None) -> BaseModel:
        expression = getattr(input, "text", str(input)).strip()
        if not expression:
            return RawText(text="No expression provided.")
        result = self._eval_expression(expression, provider)
        if result is not None:
            return RawText(text=str(result))
        return RawText(text=f"Error: cannot compute '{expression}'")

    def _eval_expression(self, expression: str, provider: Any = None) -> object | None:
        if provider is not None and hasattr(provider, "generate"):
            py_expr = self._llm_to_expr(expression, provider)
            if py_expr is not None:
                try:
                    return eval(py_expr, _SAFE_GLOBALS, {})
                except Exception:
                    pass
        for cand in [expression, expression.rstrip(".?!")]:
            try:
                return eval(cand, _SAFE_GLOBALS, {})
            except Exception:
                continue
        return None

    def _llm_to_expr(self, expression: str, provider: Any) -> str | None:
        available = sorted(k for k in _SAFE_GLOBALS if not k.startswith("_"))
        prompt = (
            "Convert the following query into a single Python expression for eval(). "
            "Only use these names: " + ", ".join(available) + ".\n\n"
            "Rules:\n"
            "- Output ONLY the expression — no quotes, no prose, no markdown.\n"
            "- Use ** for exponentiation, not ^.\n"
            "- All math functions are available directly (e.g. sqrt, factorial, sin, cos, log).\n"
            "- Constants: pi, e.\n"
            "- Do NOT use any imports, lambda, exec, or attributes.\n"
            "- If the query asks for a quantity you don't know (distances, weights, "
            "physical constants other than pi/e, facts about the world), respond with "
            "exactly the word UNKNOWN — do not guess.\n\n"
            f"Query: {expression}\n\nExpression:"
        )
        try:
            result = provider.generate(prompt, {}, None)
            expr = result.text.strip()
            if expr and expr.upper() == "UNKNOWN":
                return None
            if expr and len(expr) < 500:
                return expr
        except Exception:
            pass
        return None
