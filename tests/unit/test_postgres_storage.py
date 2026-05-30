"""Integration tests for PostgresStorage.

Run with:
    LLMASM_TEST_DB=postgresql://llmasm:llmasm@localhost:15432/llmasm \
        python -m pytest tests/unit/test_postgres_storage.py -v

Skipped automatically when LLMASM_TEST_DB is not set.
"""

from __future__ import annotations

import os

import pytest

from llmasm.graph.models import (
    Artifact,
    Checkpoint,
    Goal,
    MemoryItem,
    ModelCall,
    Run,
    RunNodeState,
    TaskEdge,
    TaskGraph,
    ToolCall,
    WorkspaceEdge,
    WorkspaceEdgeType,
    WorkspaceGraph,
    Node,
    NodeKind,
)
from llmasm.ids import new_id

TEST_DSN = os.environ.get("LLMASM_TEST_DB", "")
needs_db = pytest.mark.skipif(not TEST_DSN, reason="LLMASM_TEST_DB not set")

_TRUNCATE_TABLES = """
TRUNCATE workspace_graphs, task_graphs, nodes, task_edges, workspace_edges,
    runs, run_node_states, artifacts, goals, checkpoints,
    tool_calls, model_calls, memory_items, compilation_failures
CASCADE
"""


@pytest.fixture()
def storage():
    from llmasm.storage.postgres import PostgresStorage

    s = PostgresStorage(TEST_DSN)
    with s.conn.cursor() as cur:
        cur.execute(_TRUNCATE_TABLES)
    yield s
    with s.conn.cursor() as cur:
        cur.execute(_TRUNCATE_TABLES)
    s.conn.close()


def _workspace(name: str = "test") -> WorkspaceGraph:
    return WorkspaceGraph(id=new_id("workspace"), name=name)


def _task_graph(workspace_id: str) -> TaskGraph:
    tg_id = new_id("taskgraph")
    node = Node(
        id=new_id("node"),
        workspace_graph_id=workspace_id,
        task_graph_id=tg_id,
        kind=NodeKind.INTENT,
        name="root_intent",
        output_schema="RawText",
    )
    return TaskGraph(
        id=tg_id,
        workspace_graph_id=workspace_id,
        nodes=[node],
        metadata={"intent": "test intent", "proposal_json": '{"nodes":[]}'},
    )


def _run(workspace_id: str, tg_id: str) -> Run:
    return Run(id=new_id("run"), workspace_graph_id=workspace_id, task_graph_id=tg_id)


# ── WorkspaceGraph ────────────────────────────────────────────────────────────


@needs_db
def test_workspace_graph_round_trip(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    loaded = storage.load_workspace_graph(ws.id)
    assert loaded.id == ws.id
    assert loaded.name == ws.name
    assert loaded.status == "active"


@needs_db
def test_create_workspace_graph_idempotent(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    storage.create_workspace_graph(ws)  # ON CONFLICT DO NOTHING – must not raise
    loaded = storage.load_workspace_graph(ws.id)
    assert loaded.id == ws.id


@needs_db
def test_load_workspace_graph_missing_raises(storage):
    from llmasm.errors import StorageError

    with pytest.raises(StorageError):
        storage.load_workspace_graph("workspace_nonexistent")


# ── TaskGraph ─────────────────────────────────────────────────────────────────


@needs_db
def test_task_graph_round_trip(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    tg = _task_graph(ws.id)
    storage.persist_task_graph(tg)

    loaded = storage.load_task_graph(tg.id)
    assert loaded.id == tg.id
    assert loaded.workspace_graph_id == ws.id
    assert len(loaded.nodes) == 1
    assert loaded.nodes[0].kind == NodeKind.INTENT
    assert loaded.metadata["intent"] == "test intent"


@needs_db
def test_task_graph_persists_edges(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    tg = _task_graph(ws.id)
    n1 = tg.nodes[0]
    n2 = Node(
        id=new_id("node"),
        workspace_graph_id=ws.id,
        task_graph_id=tg.id,
        kind=NodeKind.FINAL,
        name="final",
        input_schema="Summary",
        output_schema="FinalAnswer",
    )
    edge = TaskEdge(
        id=new_id("edge"),
        workspace_graph_id=ws.id,
        task_graph_id=tg.id,
        from_node_id=n1.id,
        from_port="output",
        to_node_id=n2.id,
        to_port="input",
    )
    tg.nodes.append(n2)
    tg.task_edges.append(edge)
    storage.persist_task_graph(tg)

    loaded = storage.load_task_graph(tg.id)
    assert len(loaded.nodes) == 2
    assert len(loaded.task_edges) == 1
    assert loaded.task_edges[0].from_node_id == n1.id


# ── Run ───────────────────────────────────────────────────────────────────────


@needs_db
def test_run_create_load_update(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    tg = _task_graph(ws.id)
    storage.persist_task_graph(tg)

    run = _run(ws.id, tg.id)
    storage.create_run(run)
    loaded = storage.load_run(run.id)
    assert loaded.status == "pending"

    run.status = "running"
    storage.update_run(run)
    assert storage.load_run(run.id).status == "running"


# ── RunNodeState ──────────────────────────────────────────────────────────────


@needs_db
def test_run_node_state_create_update(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    tg = _task_graph(ws.id)
    storage.persist_task_graph(tg)
    run = _run(ws.id, tg.id)
    storage.create_run(run)

    node_id = tg.nodes[0].id
    state = RunNodeState(run_id=run.id, node_id=node_id)
    storage.create_run_node_state(state)

    states = storage.list_run_node_states(run.id)
    assert len(states) == 1
    assert states[0].status == "pending"

    state.status = "succeeded"
    state.attempts = 1
    storage.update_run_node_state(state)
    assert storage.list_run_node_states(run.id)[0].status == "succeeded"


# ── WorkspaceEdge ─────────────────────────────────────────────────────────────


@needs_db
def test_workspace_edge_persist_and_list(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    tg = _task_graph(ws.id)
    storage.persist_task_graph(tg)

    edge = WorkspaceEdge(
        id=new_id("edge"),
        workspace_graph_id=ws.id,
        edge_type=WorkspaceEdgeType.PRODUCED,
        from_type="node",
        from_id=tg.nodes[0].id,
        to_type="memory",
        to_id="memory_abc",
    )
    storage.persist_workspace_edge(edge)
    edges = storage.list_workspace_edges(ws.id)
    assert len(edges) == 1
    assert edges[0].edge_type == WorkspaceEdgeType.PRODUCED


@needs_db
def test_load_workspace_edges_for_task(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    tg = _task_graph(ws.id)
    storage.persist_task_graph(tg)

    edge = WorkspaceEdge(
        id=new_id("edge"),
        workspace_graph_id=ws.id,
        edge_type=WorkspaceEdgeType.FOLLOWS_UP,
        from_type="taskgraph",
        from_id=tg.id,
        to_type="taskgraph",
        to_id="taskgraph_prev",
    )
    storage.persist_workspace_edge(edge)
    edges = storage.load_workspace_edges_for_task(tg.id)
    assert any(e.id == edge.id for e in edges)


# ── Artifact ──────────────────────────────────────────────────────────────────


@needs_db
def test_artifact_persist_and_load(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    tg = _task_graph(ws.id)
    storage.persist_task_graph(tg)
    run = _run(ws.id, tg.id)
    storage.create_run(run)

    artifact = Artifact(
        id=new_id("artifact"),
        run_id=run.id,
        node_id=tg.nodes[0].id,
        port="output",
        content_json={"text": "hello"},
        token_count=5,
    )
    storage.persist_artifact(artifact)
    loaded = storage.load_artifact(artifact.id)
    assert loaded.content_json == {"text": "hello"}

    listed = storage.list_artifacts(run.id)
    assert len(listed) == 1


# ── Goal ──────────────────────────────────────────────────────────────────────


@needs_db
def test_goal_lifecycle(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)

    goal = Goal(id=new_id("goal"), workspace_graph_id=ws.id, text="answer Q", status="active")
    storage.persist_goal(goal)

    active = storage.load_active_goal(ws.id)
    assert active is not None
    assert active.text == "answer Q"

    goal.status = "closed"
    storage.update_goal(goal)
    assert storage.load_active_goal(ws.id) is None
    assert storage.load_goal(goal.id).status == "closed"


# ── MemoryItem & context retrieval ────────────────────────────────────────────


@needs_db
def test_memory_items_persist_and_list(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)

    items = [
        MemoryItem(
            id=new_id("memory"),
            workspace_graph_id=ws.id,
            kind="turn",
            text=f"Q: question {i}? A: answer {i}",
        )
        for i in range(3)
    ]
    for item in items:
        storage.persist_memory_item(item)

    listed = storage.list_memory_items(ws.id)
    assert len(listed) == 3


@needs_db
def test_retrieve_workspace_context_returns_scored_items(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)

    storage.persist_memory_item(
        MemoryItem(
            id=new_id("memory"),
            workspace_graph_id=ws.id,
            kind="turn",
            text="Q: capital of France? A: Paris",
        )
    )
    storage.persist_memory_item(
        MemoryItem(
            id=new_id("memory"),
            workspace_graph_id=ws.id,
            kind="turn",
            text="Q: best pizza? A: Napoli",
        )
    )

    ctx = storage.retrieve_workspace_context(ws.id, "capital france", 1000)
    assert len(ctx) >= 1
    assert "Paris" in ctx[0].text


# ── Few-shot examples ─────────────────────────────────────────────────────────


@needs_db
def test_retrieve_few_shot_examples(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    tg = _task_graph(ws.id)
    storage.persist_task_graph(tg)

    examples = storage.retrieve_few_shot_examples(ws.id, "test intent", 5)
    assert len(examples) == 1
    assert examples[0].intent == "test intent"
    assert examples[0].task_graph_id == tg.id


# ── Checkpoint ────────────────────────────────────────────────────────────────


@needs_db
def test_checkpoint_persist_and_list(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    tg = _task_graph(ws.id)
    storage.persist_task_graph(tg)
    run = _run(ws.id, tg.id)
    storage.create_run(run)

    cp = Checkpoint(
        id=new_id("checkpoint"),
        run_id=run.id,
        state_hash="abc123",
        completed_node_ids=[tg.nodes[0].id],
    )
    storage.persist_checkpoint(cp)
    listed = storage.list_checkpoints(run.id)
    assert len(listed) == 1
    assert listed[0].state_hash == "abc123"
    assert tg.nodes[0].id in listed[0].completed_node_ids


# ── ToolCall / ModelCall ──────────────────────────────────────────────────────


@needs_db
def test_tool_and_model_calls(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    tg = _task_graph(ws.id)
    storage.persist_task_graph(tg)
    run = _run(ws.id, tg.id)
    storage.create_run(run)

    tc = ToolCall(
        id=new_id("artifact"),
        run_id=run.id,
        node_id=tg.nodes[0].id,
        tool_name="search",
        input_json={"text": "query"},
        status="succeeded",
        latency_ms=42,
    )
    storage.persist_tool_call(tc)
    assert len(storage.list_tool_calls(run.id)) == 1

    mc = ModelCall(
        id=new_id("artifact"),
        run_id=run.id,
        node_id=tg.nodes[0].id,
        provider="ollama",
        model="llama3.1:8b",
        status="succeeded",
    )
    storage.persist_model_call(mc)
    assert len(storage.list_model_calls(run.id)) == 1


# ── Compilation failure ───────────────────────────────────────────────────────


@needs_db
def test_persist_compilation_failure(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    storage.persist_compilation_failure(ws.id, {"prompt": "x", "errors": "oops"})
    # No exception = pass; failures table has BIGSERIAL PK so no uniqueness issue


# ── find_cached_artifact ──────────────────────────────────────────────────────


@needs_db
def test_find_cached_artifact(storage):
    ws = _workspace()
    storage.create_workspace_graph(ws)
    tg = _task_graph(ws.id)
    storage.persist_task_graph(tg)
    run = _run(ws.id, tg.id)
    storage.create_run(run)

    art = Artifact(
        id=new_id("artifact"),
        run_id=run.id,
        node_id=tg.nodes[0].id,
        port="output",
        content_json={"text": "cached"},
        metadata={"cache_key": "key_abc", "input_artifact_ids": ["art_1", "art_2"]},
    )
    storage.persist_artifact(art)

    found = storage.find_cached_artifact("key_abc", ["art_2", "art_1"])
    assert found is not None
    assert found.id == art.id

    not_found = storage.find_cached_artifact("key_abc", ["art_1"])
    assert not_found is None

    not_found2 = storage.find_cached_artifact("key_xyz", ["art_1", "art_2"])
    assert not_found2 is None
