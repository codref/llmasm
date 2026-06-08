"""Planner output parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError as PydanticValidationError

from llmasm.compiler.proposal import TaskGraphProposal
from llmasm.graph.validation import ValidationIssue


@dataclass(frozen=True)
class ParseResult:
    """Parsed proposal or validation issues."""

    proposal: TaskGraphProposal | None
    issues: list[ValidationIssue]


def parse_task_graph_proposal(raw: str) -> ParseResult:
    """Parse planner JSON into a proposal model."""

    try:
        return ParseResult(TaskGraphProposal.model_validate_json(_extract_json(raw)), [])
    except (json.JSONDecodeError, PydanticValidationError) as exc:
        return ParseResult(None, [ValidationIssue("PARSE_FAILURE", None, str(exc))])


def _extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text
