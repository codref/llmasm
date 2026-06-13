"""Regression tests for planner port-direction handling.

A planner (Ollama) emitted ``"direction": "in"`` instead of ``"input"``,
which the literal-typed ``Port`` model rejected with a Pydantic
ValidationError that propagated out of the repair loop and crashed the run.

These tests pin down the two layers of defense added to fix that crash:
- ``ProposalPort.direction`` is now a ``Literal["input", "output"]`` so
  the parser surfaces bad values as a recoverable ``PARSE_FAILURE`` issue.
- ``_canonical_port_direction`` is a defense-in-depth helper that maps the
  common ``in``/``out`` shorthand to the canonical values.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError as PydanticValidationError

from llmasm.compiler.compiler import (
    Compiler,
    _canonical_port_direction,
)
from llmasm.compiler.parser import parse_task_graph_proposal
from llmasm.compiler.proposal import ProposalNode, ProposalPort, TaskGraphProposal
from llmasm.config import RuntimeConfig
from llmasm.errors import CompilationError
from llmasm.graph.models import WorkspaceGraph
from llmasm.graph.registry import default_schema_registry
from llmasm.graph.transforms import default_transform_registry
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.registry import ToolRegistry
from tests.unit.fakes import FakeProvider


# ---------------------------------------------------------------------------
# Unit tests for the coercion helper (Layer 1).
# ---------------------------------------------------------------------------


def test_canonical_port_direction_accepts_canonical_values() -> None:
    assert _canonical_port_direction("input") == "input"
    assert _canonical_port_direction("output") == "output"


def test_canonical_port_direction_accepts_common_shorthand() -> None:
    assert _canonical_port_direction("in") == "input"
    assert _canonical_port_direction("out") == "output"


def test_canonical_port_direction_rejects_unknown() -> None:
    assert _canonical_port_direction("sideways") is None
    assert _canonical_port_direction("") is None
    assert _canonical_port_direction("INPUT") is None  # case-sensitive


# ---------------------------------------------------------------------------
# Layer 2: ProposalPort enforces the literal at the model level.
# ---------------------------------------------------------------------------


def test_proposal_port_accepts_canonical_direction() -> None:
    ProposalPort(name="in", direction="input", schema_ref="RawText")
    ProposalPort(name="out", direction="output", schema_ref="RawText")


def test_proposal_port_rejects_shorthand_direction() -> None:
    with pytest.raises(PydanticValidationError):
        ProposalPort(name="in", direction="in", schema_ref="RawText")
    with pytest.raises(PydanticValidationError):
        ProposalPort(name="out", direction="out", schema_ref="RawText")


def test_proposal_port_rejects_unknown_direction() -> None:
    with pytest.raises(PydanticValidationError):
        ProposalPort(name="x", direction="sideways", schema_ref="RawText")


# ---------------------------------------------------------------------------
# The parser must surface invalid direction values as a recoverable issue.
# ---------------------------------------------------------------------------


def test_parser_surfaces_shorthand_direction_as_parse_failure() -> None:
    """A planner that emits 'in'/'out' must be caught at parse time, not crash."""

    raw = json.dumps(
        {
            "intent": "answer question",
            "goal_action": "new",
            "nodes": [
                {
                    "name": "answer",
                    "kind": "model",
                    "input_schema": "RawText",
                    "output_schema": "Summary",
                    "execution": {
                        "provider": "fake",
                        "model": "fake-model",
                        "allow_cache": False,
                    },
                    "ports": [
                        {
                            "name": "in",
                            "direction": "in",
                            "schema_ref": "RawText",
                        }
                    ],
                },
                {
                    "name": "final",
                    "kind": "final",
                    "input_schema": "Summary",
                    "output_schema": "FinalAnswer",
                },
            ],
            "edges": [
                {
                    "from_node": "answer",
                    "from_port": "out",
                    "to_node": "final",
                    "to_port": "in",
                }
            ],
        }
    )
    parsed = parse_task_graph_proposal(raw)
    assert parsed.proposal is None
    assert len(parsed.issues) == 1
    assert parsed.issues[0].code == "PARSE_FAILURE"
    # The Pydantic error should mention the offending field.
    assert "direction" in parsed.issues[0].detail


def test_parser_accepts_canonical_direction() -> None:
    raw = json.dumps(
        {
            "intent": "answer question",
            "goal_action": "new",
            "nodes": [
                {
                    "name": "answer",
                    "kind": "model",
                    "input_schema": "RawText",
                    "output_schema": "Summary",
                    "execution": {
                        "provider": "fake",
                        "model": "fake-model",
                        "allow_cache": False,
                    },
                    "ports": [
                        {
                            "name": "in",
                            "direction": "input",
                            "schema_ref": "RawText",
                        }
                    ],
                },
                {
                    "name": "final",
                    "kind": "final",
                    "input_schema": "Summary",
                    "output_schema": "FinalAnswer",
                },
            ],
            "edges": [
                {
                    "from_node": "answer",
                    "from_port": "output",
                    "to_node": "final",
                    "to_port": "input",
                }
            ],
        }
    )
    parsed = parse_task_graph_proposal(raw)
    assert parsed.proposal is not None
    assert parsed.issues == []


# ---------------------------------------------------------------------------
# JSON-schema constraint — used by the planner provider for structured output.
# ---------------------------------------------------------------------------


def _find_port_direction_enum(schema: dict) -> list | None:
    """Walk the JSON schema and return the 'direction' enum on a ProposalPort, if present."""

    def visit(node: object) -> list | None:
        if isinstance(node, dict):
            if "$ref" in node:
                return None
            if node.get("type") == "object" and "direction" in node.get("properties", {}):
                direction = node["properties"]["direction"]
                if "enum" in direction:
                    return direction["enum"]
            for value in node.values():
                found = visit(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = visit(item)
                if found is not None:
                    return found
        return None

    return visit(schema)


def test_proposal_json_schema_constrains_direction() -> None:
    schema = TaskGraphProposal.model_json_schema()
    port_schema = _find_port_direction_enum(schema)
    assert port_schema is not None
    assert sorted(port_schema) == ["input", "output"]


# ---------------------------------------------------------------------------
# _normalize must accept a proposal with explicit input/output ports.
# ---------------------------------------------------------------------------


def _build_compiler(
    storage: InMemoryStorage | None = None,
    planner: FakeProvider | None = None,
) -> Compiler:
    return Compiler(
        storage=storage or InMemoryStorage(),
        planner=planner or FakeProvider(),
        schema_registry=default_schema_registry(),
        transform_registry=default_transform_registry(),
        tool_registry=ToolRegistry(default_schema_registry()),
        runtime_config=RuntimeConfig(),
    )


def test_normalize_accepts_proposal_with_explicit_input_port() -> None:
    compiler = _build_compiler()
    proposal = TaskGraphProposal(
        intent="answer",
        goal_action="new",
        nodes=[
            ProposalNode(
                name="answer",
                kind="model",
                input_schema="RawText",
                output_schema="Summary",
                execution={
                    "provider": "fake",
                    "model": "fake-model",
                    "allow_cache": False,
                },
                ports=[ProposalPort(name="in", direction="input", schema_ref="RawText")],
            ),
            ProposalNode(
                name="final",
                kind="final",
                input_schema="Summary",
                output_schema="FinalAnswer",
            ),
        ],
        edges=[],
    )
    graph, issues = compiler._normalize("workspace_test", proposal, "test", None)
    assert all(i.code != "PORT_DIRECTION_INVALID" for i in issues)
    answer_node = next(n for n in graph.nodes if n.name == "answer")
    in_port = next(p for p in answer_node.ports if p.name == "in")
    assert in_port.direction == "input"


# ---------------------------------------------------------------------------
# End-to-end: a planner that emits the shorthand must not crash the run.
# Reproduces the session23 Q22 crash scenario.
# ---------------------------------------------------------------------------


def test_repair_loop_recovers_from_shorthand_direction() -> None:
    """Compile a proposal whose first attempt uses 'in'/'out', then a valid one.

    Exercises the full repair loop: first attempt fails to parse, repair
    loop retries, second attempt succeeds.
    """
    storage = InMemoryStorage()
    storage.create_workspace_graph(WorkspaceGraph(id="workspace_e2e", name="e2e"))

    bad_proposal = json.dumps(
        {
            "intent": "answer",
            "goal_action": "new",
            "nodes": [
                {
                    "name": "answer",
                    "kind": "model",
                    "input_schema": "RawText",
                    "output_schema": "Summary",
                    "ports": [
                        {
                            "name": "in",
                            "direction": "in",
                            "schema_ref": "RawText",
                        }
                    ],
                },
                {
                    "name": "final",
                    "kind": "final",
                    "input_schema": "Summary",
                    "output_schema": "FinalAnswer",
                },
            ],
            "edges": [],
        }
    )
    good_proposal = json.dumps(
        {
            "intent": "answer",
            "goal_action": "new",
            "nodes": [
                {
                    "name": "answer",
                    "kind": "model",
                    "input_schema": "RawText",
                    "output_schema": "Summary",
                    "execution": {
                        "provider": "fake",
                        "model": "fake-model",
                        "allow_cache": False,
                    },
                },
                {
                    "name": "final",
                    "kind": "final",
                    "input_schema": "Summary",
                    "output_schema": "FinalAnswer",
                },
            ],
            "edges": [],
        }
    )

    provider = FakeProvider(planner_outputs=[bad_proposal, good_proposal])
    compiler = _build_compiler(storage=storage, planner=provider)

    # Sanity: each proposal must parse in isolation.
    parsed_bad = parse_task_graph_proposal(bad_proposal)
    parsed_good = parse_task_graph_proposal(good_proposal)
    assert parsed_bad.proposal is None  # shorthand fails the literal
    assert parsed_good.proposal is not None  # canonical parses fine

    # The bad proposal would have crashed in the legacy code path. The
    # current path must convert it into a recoverable ValidationIssue and
    # accept the second attempt.
    try:
        task_graph_id = compiler.compile_into_workspace("workspace_e2e", "answer my question")
    except CompilationError as exc:  # pragma: no cover - defensive
        pytest.fail(
            f"compile_into_workspace raised CompilationError after {exc.attempts} attempts: "
            f"{[(e.code, e.detail[:80]) for e in exc.last_errors]}"
        )
    assert task_graph_id.startswith("taskgraph_")
    graph = storage.load_task_graph(task_graph_id)
    assert any(node.name == "answer" for node in graph.nodes)
    assert any(node.name == "final" for node in graph.nodes)
