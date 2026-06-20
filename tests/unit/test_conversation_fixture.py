"""Tests for the ConversationFixture runner."""

from __future__ import annotations

import json
from pathlib import Path


from llmasm.api import LLMASM
from llmasm.config import RuntimeConfig
from llmasm.graph.models import WorkspaceEdgeType
from llmasm.graph.registry import default_schema_registry
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.registry import ToolRegistry
from tests.fixtures.conversation_runner import (
    ConversationFixture,
    TurnSpec,
    run_fixture,
)
from tests.unit.fakes import ConversationRetrieveTool, FakeProvider, conversation_summary_proposal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(planner_outputs: list[str], model_text: str = "summary text") -> tuple[LLMASM, InMemoryStorage]:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    tools.register(ConversationRetrieveTool())
    provider = FakeProvider(planner_outputs=planner_outputs, model_text=model_text)
    storage = InMemoryStorage()
    app = LLMASM(
        storage=storage,
        provider=provider,
        tool_registry=tools,
        runtime_config=RuntimeConfig(default_model="fake-model"),
        schema_registry=schemas,
    )
    return app, storage


def _new_turn_proposal() -> str:
    """Minimal planner proposal used for each test turn (goal_action=new)."""
    return conversation_summary_proposal()


# ---------------------------------------------------------------------------
# TurnSpec / ConversationFixture dataclass tests
# ---------------------------------------------------------------------------


def test_turn_spec_round_trips_json() -> None:
    spec = TurnSpec(
        prompt="what is the plan?",
        inject_memory=["context note"],
        expected_keywords=["plan", "scope"],
    )
    restored = TurnSpec.from_dict(spec.to_dict())
    assert restored == spec


def test_fixture_round_trips_json(tmp_path: Path) -> None:
    fixture = ConversationFixture(
        name="test",
        description="a test",
        turns=[TurnSpec(prompt="hello"), TurnSpec(prompt="world")],
    )
    path = tmp_path / "fixture.json"
    fixture.to_json(path)
    restored = ConversationFixture.from_json(path)
    assert restored.name == "test"
    assert len(restored.turns) == 2
    assert restored.turns[0].prompt == "hello"


def test_fixture_from_json_defaults() -> None:
    """TurnSpec fields have sane defaults when absent from JSON."""
    raw = json.dumps({"name": "x", "turns": [{"prompt": "ask"}]})
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(raw)
        name = f.name
    fixture = ConversationFixture.from_json(Path(name))
    os.unlink(name)
    assert fixture.turns[0].inject_memory == []
    assert fixture.turns[0].expected_keywords == []


# ---------------------------------------------------------------------------
# run_fixture mechanics
# ---------------------------------------------------------------------------


def test_run_fixture_single_turn_produces_memory_and_edges() -> None:
    app, storage = _make_app([_new_turn_proposal()])
    workspace_id = app.create_workspace("test")
    fixture = ConversationFixture(name="single", turns=[
        TurnSpec(prompt="new task: summarize the conversation", expected_keywords=["summary"]),
    ])

    results = run_fixture(app, workspace_id, fixture)

    assert len(results) == 1
    result = results[0]
    assert result.turn_index == 0
    assert result.goal_action == "new"
    assert result.keywords_passed
    assert not result.missing_keywords

    # One turn MemoryItem persisted
    items = storage.list_memory_items(workspace_id)
    assert len(items) == 1
    assert items[0].kind == "turn"

    # One PRODUCED edge (no FOLLOWS_UP on first turn)
    edges = storage.list_workspace_edges(workspace_id)
    produced = [e for e in edges if e.edge_type == WorkspaceEdgeType.PRODUCED]
    follows = [e for e in edges if e.edge_type == WorkspaceEdgeType.FOLLOWS_UP]
    assert len(produced) == 1
    assert len(follows) == 0


def test_run_fixture_three_turns_edge_and_memory_counts() -> None:
    app, storage = _make_app([_new_turn_proposal()] * 3)
    workspace_id = app.create_workspace("test")
    fixture = ConversationFixture(name="three", turns=[
        TurnSpec(prompt="new task: first question"),
        TurnSpec(prompt="new task: second question"),
        TurnSpec(prompt="new task: third question"),
    ])

    results = run_fixture(app, workspace_id, fixture)

    assert len(results) == 3

    items = storage.list_memory_items(workspace_id)
    turn_items = [i for i in items if i.kind == "turn"]
    assert len(turn_items) == 3

    edges = storage.list_workspace_edges(workspace_id)
    produced = [e for e in edges if e.edge_type == WorkspaceEdgeType.PRODUCED]
    follows = [e for e in edges if e.edge_type == WorkspaceEdgeType.FOLLOWS_UP]
    # N PRODUCED edges, (N-1) FOLLOWS_UP edges
    assert len(produced) == 3
    assert len(follows) == 2


def test_run_fixture_injected_memory_persisted_before_turn() -> None:
    app, storage = _make_app([_new_turn_proposal()] * 2)
    workspace_id = app.create_workspace("test")
    fixture = ConversationFixture(name="inject", turns=[
        TurnSpec(prompt="new task: first"),
        TurnSpec(
            prompt="new task: second",
            inject_memory=["the user prefers concise answers", "focus on storage layer"],
        ),
    ])

    results = run_fixture(app, workspace_id, fixture)

    assert len(results) == 2
    items = storage.list_memory_items(workspace_id)
    human_notes = [i for i in items if i.kind == "human_note"]
    assert len(human_notes) == 2
    assert human_notes[0].text == "the user prefers concise answers"
    assert human_notes[1].text == "focus on storage layer"
    # metadata records the turn index they were injected before
    assert human_notes[0].metadata["injected_before_turn"] == 1
    assert human_notes[1].metadata["injected_before_turn"] == 1


def test_run_fixture_keyword_failure_reported() -> None:
    app, storage = _make_app([_new_turn_proposal()])
    workspace_id = app.create_workspace("test")
    fixture = ConversationFixture(name="kw", turns=[
        TurnSpec(
            prompt="new task: summarize",
            expected_keywords=["summary", "KEYWORD_ABSENT_FROM_RESPONSE"],
        ),
    ])

    results = run_fixture(app, workspace_id, fixture)

    assert len(results) == 1
    assert not results[0].keywords_passed
    assert "KEYWORD_ABSENT_FROM_RESPONSE".lower() in [k.lower() for k in results[0].missing_keywords]


def test_follows_up_edges_chain_task_graphs() -> None:
    """FOLLOWS_UP edges link consecutive task graphs in order."""
    app, storage = _make_app([_new_turn_proposal()] * 3)
    workspace_id = app.create_workspace("test")
    fixture = ConversationFixture(name="chain", turns=[
        TurnSpec(prompt="new task: one"),
        TurnSpec(prompt="new task: two"),
        TurnSpec(prompt="new task: three"),
    ])

    results = run_fixture(app, workspace_id, fixture)

    edges = storage.list_workspace_edges(workspace_id)
    follows = [e for e in edges if e.edge_type == WorkspaceEdgeType.FOLLOWS_UP]
    # Turn 1 follows Turn 0, Turn 2 follows Turn 1
    follows_by_from = {e.from_id: e for e in follows}
    assert results[1].task_graph_id in follows_by_from
    assert follows_by_from[results[1].task_graph_id].to_id == results[0].task_graph_id
    assert results[2].task_graph_id in follows_by_from
    assert follows_by_from[results[2].task_graph_id].to_id == results[1].task_graph_id


def test_sample_fixture_file_loads() -> None:
    """The committed sample_chat.json is valid and loads without error."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_chat.json"
    fixture = ConversationFixture.from_json(fixture_path)
    assert fixture.name == "context_aware_qa"
    assert len(fixture.turns) == 3
    assert fixture.turns[2].inject_memory  # third turn has an injection
