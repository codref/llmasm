"""Compiler repair loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from llmasm.compiler.parser import parse_task_graph_proposal
from llmasm.compiler.proposal import TaskGraphProposal
from llmasm.errors import CompilationError
from llmasm.graph.validation import ValidationIssue


class Planner(Protocol):
    """Protocol for planner model calls."""

    def generate(
        self,
        prompt: str,
        options: dict[str, Any] | None = None,
        format_schema: dict[str, Any] | None = None,
    ) -> object:
        """Generate planner output."""


def compile_with_repair(
    planner: Planner,
    planner_prompt: str,
    expected_goal_action: str,
    max_attempts: int,
    validator: Callable[[TaskGraphProposal, str], list[ValidationIssue]],
    planner_options: dict[str, Any] | None = None,
) -> TaskGraphProposal:
    """Call planner, parse output, validate, and retry with issue feedback."""

    prompt = planner_prompt
    last_errors: list[ValidationIssue] = []
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        output = planner.generate(
            prompt,
            planner_options or {},
            TaskGraphProposal.model_json_schema(),
        )
        last_raw = str(getattr(output, "text", output))
        parsed = parse_task_graph_proposal(last_raw)
        if parsed.proposal is None:
            last_errors = parsed.issues
        else:
            last_errors = validator(parsed.proposal, expected_goal_action)
            if not last_errors:
                return parsed.proposal
        prompt = (
            planner_prompt
            + "\n\nYour previous attempt had errors. Fix ONLY the issues listed below; "
            "keep all other nodes and edges unchanged.\n"
            "Previous proposal:\n"
            + last_raw
            + "\n\nErrors to fix:\n"
            + _format_issues(last_errors)
        )
    raise CompilationError(
        "Planner failed to produce a valid task graph",
        attempts=max_attempts,
        last_errors=last_errors,
        last_raw_output=last_raw,
    )


def _format_issues(issues: list[ValidationIssue]) -> str:
    return "\n".join(f"- {issue.code}: {issue.detail}" for issue in issues)
