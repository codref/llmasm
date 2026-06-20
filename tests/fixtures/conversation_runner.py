"""Conversation fixture runner for multi-turn regression tests.

A ConversationFixture is a scripted sequence of turns that can be
loaded from JSON and replayed against any LLMASM app instance.
Each turn can pre-inject MemoryItems before compilation, simulating
man-in-the-middle context additions.  After execution the runner
persists FOLLOWS_UP and PRODUCED workspace edges using the same
pattern as examples/chat.py.

Usage::

    fixture = ConversationFixture.from_json(Path("tests/fixtures/sample_chat.json"))
    results = run_fixture(app, workspace_id, fixture)
    assert all(r.keywords_passed for r in results)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmasm.api import LLMASM
from llmasm.analysis.run import RunAnalysis
from llmasm.graph.models import MemoryItem, WorkspaceEdge, WorkspaceEdgeType
from llmasm.ids import new_id
from llmasm.schemas import FinalAnswer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TurnSpec:
    """Specification for one conversation turn.

    Attributes:
        prompt: The user prompt sent to the compiler.
        inject_memory: Texts to persist as ``human_note`` MemoryItems
            *before* this turn compiles.  Simulates a human injecting
            context between turns.
        expected_keywords: Case-insensitive substrings that must all
            appear somewhere in the answer text for the turn to pass.
    """

    prompt: str
    inject_memory: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "inject_memory": self.inject_memory,
            "expected_keywords": self.expected_keywords,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnSpec:
        return cls(
            prompt=data["prompt"],
            inject_memory=data.get("inject_memory", []),
            expected_keywords=data.get("expected_keywords", []),
        )


@dataclass
class ConversationFixture:
    """A named multi-turn conversation script.

    Attributes:
        name: Human-readable fixture name.
        description: Optional longer description.
        turns: Ordered list of turn specifications.
    """

    name: str
    turns: list[TurnSpec]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "turns": [t.to_dict() for t in self.turns],
        }

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationFixture:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            turns=[TurnSpec.from_dict(t) for t in data["turns"]],
        )

    @classmethod
    def from_json(cls, path: Path) -> ConversationFixture:
        return cls.from_dict(json.loads(path.read_text()))


@dataclass
class TurnResult:
    """Outcome of executing one fixture turn.

    Attributes:
        turn_index: Zero-based index within the fixture.
        prompt: The user prompt used.
        answer_text: Text of the FinalAnswer produced.
        task_graph_id: ID of the compiled task graph.
        run_id: ID of the execution run.
        goal_action: Goal action string from task graph metadata.
        memory_items_created: IDs of MemoryItems persisted this turn.
        keywords_passed: True when all expected_keywords appear in answer_text.
        missing_keywords: Keywords from expected_keywords absent from the answer.
    """

    turn_index: int
    prompt: str
    answer_text: str
    task_graph_id: str
    run_id: str
    goal_action: str
    memory_items_created: list[str]
    keywords_passed: bool
    missing_keywords: list[str]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_fixture(
    app: LLMASM,
    workspace_id: str,
    fixture: ConversationFixture,
) -> list[TurnResult]:
    """Execute a ConversationFixture against *app* inside *workspace_id*.

    For each turn:
    1. Inject any pre-specified MemoryItems (human_note kind).
    2. Compile the prompt into a task graph.
    3. Run the task graph.
    4. Persist the answer as a ``turn`` MemoryItem.
    5. Write FOLLOWS_UP and PRODUCED workspace edges.
    6. Assert expected_keywords against the answer text.

    Returns a list of TurnResult, one per turn, in order.
    """
    storage = app.storage
    results: list[TurnResult] = []
    previous_task_graph_id: str | None = None

    for i, turn_spec in enumerate(fixture.turns):
        # Step 1: inject context before compilation
        for note in turn_spec.inject_memory:
            storage.persist_memory_item(
                MemoryItem(
                    id=new_id("memory"),
                    workspace_graph_id=workspace_id,
                    kind="human_note",
                    text=note,
                    metadata={"fixture": fixture.name, "injected_before_turn": i},
                )
            )

        # Step 2-3: compile and run
        task_graph_id = app.compile(workspace_id, turn_spec.prompt)
        run_id = app.run(task_graph_id)
        analysis = app.query_run(run_id)

        # Step 4: extract answer
        final_id = _final_node_id(analysis)
        answer = _extract_answer(analysis, final_id)
        goal_action = analysis.task_graph.metadata.get("goal_action", "")

        # Step 5: persist turn MemoryItem
        memory = MemoryItem(
            id=new_id("memory"),
            workspace_graph_id=workspace_id,
            kind="turn",
            text=f"Q: {turn_spec.prompt}\nA: {answer.text}",
            source_run_id=run_id,
            metadata={"fixture": fixture.name, "turn": i, "goal_action": goal_action},
        )
        storage.persist_memory_item(memory)
        memory_ids = [memory.id]

        # Step 5b: workspace edges
        if previous_task_graph_id is not None:
            storage.persist_workspace_edge(
                WorkspaceEdge(
                    id=new_id("edge"),
                    workspace_graph_id=workspace_id,
                    edge_type=WorkspaceEdgeType.FOLLOWS_UP,
                    from_type="task_graph",
                    from_id=task_graph_id,
                    to_type="task_graph",
                    to_id=previous_task_graph_id,
                    reason=f"fixture turn {i} follows turn {i - 1}",
                )
            )

        if final_id is not None:
            storage.persist_workspace_edge(
                WorkspaceEdge(
                    id=new_id("edge"),
                    workspace_graph_id=workspace_id,
                    edge_type=WorkspaceEdgeType.PRODUCED,
                    from_type="node",
                    from_id=final_id,
                    to_type="memory_item",
                    to_id=memory.id,
                    reason="final node answer stored as workspace memory",
                )
            )

        previous_task_graph_id = task_graph_id

        # Step 6: keyword assertions
        answer_lower = answer.text.lower()
        missing = [kw for kw in turn_spec.expected_keywords if kw.lower() not in answer_lower]

        results.append(
            TurnResult(
                turn_index=i,
                prompt=turn_spec.prompt,
                answer_text=answer.text,
                task_graph_id=task_graph_id,
                run_id=run_id,
                goal_action=goal_action,
                memory_items_created=memory_ids,
                keywords_passed=not missing,
                missing_keywords=missing,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Helpers (mirrors chat.py)
# ---------------------------------------------------------------------------


def _final_node_id(analysis: RunAnalysis) -> str | None:
    for node in analysis.task_graph.nodes:
        if node.kind == "final":
            return node.id
    return None


def _extract_answer(analysis: RunAnalysis, final_id: str | None) -> FinalAnswer:
    if final_id is None:
        return FinalAnswer(text="", sources=[])
    artifacts = [a for a in analysis.artifacts if a.node_id == final_id]
    if not artifacts or not artifacts[-1].content_json:
        return FinalAnswer(text="", sources=[])
    return FinalAnswer.model_validate(artifacts[-1].content_json)
