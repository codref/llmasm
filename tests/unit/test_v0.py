from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError
from pydantic import BaseModel

from llmasm.api import LLMASM
from llmasm.analysis.visualize import to_dot, to_mermaid, to_viewer_graph
from llmasm.compiler.parser import parse_task_graph_proposal
from llmasm.config import RuntimeConfig
from llmasm.goals.classifier import classify_goal_action, classify_goal_action_llm, GoalClassification
from llmasm.runtime.context import ContextRelevanceFilter, filter_context_with_llm
from llmasm.storage.base import ContextItem
from llmasm.graph.models import Goal, Node, NodeKind, Run, RunStatus, TaskGraph, WorkspaceGraph
from llmasm.graph.registry import default_schema_registry
from llmasm.graph.transforms import default_transform_registry
from llmasm.graph.validation import validate_required_ports, validate_tools
from llmasm.ids import new_id
from llmasm.runtime.executor import Executor
from llmasm.schemas import ConversationRecord, ConversationText, FinalAnswer
from llmasm.storage.embeddings import InMemoryEmbeddingStore, NullEmbeddingStore, embed_and_persist
from llmasm.tools.base import ToolSpec
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.registry import ToolRegistry
from tests.unit.fakes import ConversationRetrieveTool, FakeProvider, conversation_summary_proposal


class RawSearchTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="raw.search",
            description="Search raw text",
            input_schema="RawText",
            output_schema="RawText",
        )

    def invoke(self, input: BaseModel) -> BaseModel:
        return input


def test_import_and_ids() -> None:
    import llmasm

    assert llmasm.__version__ == "0.1.0"
    assert new_id("workspace").startswith("workspace_")


def test_model_validation_rejects_invalid_enum() -> None:
    with pytest.raises(PydanticValidationError):
        Node(
            id="node_bad",
            workspace_graph_id="workspace_1",
            task_graph_id="taskgraph_1",
            kind="bad",
            name="bad",
        )


def test_schema_transform_extract_text() -> None:
    transforms = default_transform_registry()
    output = transforms.apply(
        "extract_text",
        ConversationRecord(id="c1", text="hello", metadata={}),
    )
    assert output == ConversationText(text="hello")


def test_parser_accepts_fenced_json() -> None:
    raw = """
```json
{
  "intent": "answer question",
  "goal_action": "new",
  "nodes": [
    {"name": "intent", "kind": "intent", "output_schema": "RawText"},
    {"name": "final", "kind": "final", "input_schema": "Summary", "output_schema": "FinalAnswer"}
  ],
  "edges": []
}
```
"""
    parsed = parse_task_graph_proposal(raw)
    assert parsed.proposal is not None
    assert parsed.proposal.intent == "answer question"


def test_validation_rejects_inputless_tool_node() -> None:
    graph = TaskGraph(
        id="taskgraph_1",
        workspace_graph_id="workspace_1",
        nodes=[
            Node(
                id="node_tool",
                workspace_graph_id="workspace_1",
                task_graph_id="taskgraph_1",
                kind=NodeKind.TOOL,
                name="retrieve",
                execution={"tool": "conversation_store.retrieve"},
            )
        ],
    )

    issues = validate_required_ports(graph)

    assert any(issue.code == "PORT_UNSATISFIED" for issue in issues)


def test_validation_rejects_tool_schema_mismatch() -> None:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    tools.register(ConversationRetrieveTool())
    graph = TaskGraph(
        id="taskgraph_1",
        workspace_graph_id="workspace_1",
        nodes=[
            Node(
                id="node_tool",
                workspace_graph_id="workspace_1",
                task_graph_id="taskgraph_1",
                kind=NodeKind.TOOL,
                name="retrieve",
                input_schema=None,
                output_schema=None,
                execution={"tool": "conversation_store.retrieve"},
            )
        ],
    )

    issues = validate_tools(graph, tools)

    assert [issue.code for issue in issues] == ["TOOL_SCHEMA_MISMATCH", "TOOL_SCHEMA_MISMATCH"]


def test_goal_classifier() -> None:
    assert classify_goal_action("new task: summarize this", None) == "new"
    active = Goal(id="goal_1", workspace_graph_id="workspace_1", text="summarize conversation")
    assert classify_goal_action("now check the weather", active) == "continue"
    assert classify_goal_action("actually focus on weather", active) == "steer"
    assert classify_goal_action("new task unrelated", active) == "new"


def test_goal_classifier_llm_happy_path() -> None:
    """LLM classifier returns the action emitted by the provider."""
    import json

    active = Goal(id="goal_1", workspace_graph_id="workspace_1", text="understand neural networks")
    # Provider returns a valid GoalClassification JSON
    provider = FakeProvider(
        planner_outputs=[
            json.dumps({"action": "continue", "reason": "This is a follow-up question."})
        ]
    )
    result = classify_goal_action_llm(
        "Can you explain backpropagation in more detail?",
        active,
        provider,
        {"model": "fake-model"},
    )
    assert result == "continue"


def test_goal_classifier_llm_fallback_on_bad_json() -> None:
    """LLM classifier falls back to deterministic classifier when provider returns garbage."""
    active = Goal(id="goal_1", workspace_graph_id="workspace_1", text="summarize conversation")
    provider = FakeProvider(planner_outputs=["not valid json at all !!!"])
    # Deterministic fallback: "new task" signal → NEW
    result = classify_goal_action_llm("new task: something unrelated", active, provider)
    assert result == "new"


def _make_ctx_item(id: str, text: str) -> ContextItem:
    return ContextItem(id=id, kind="memory_item", text=text, score=0.5, token_count=10, item=object())


def test_llm_context_filter_happy_path() -> None:
    """LLM context filter keeps only the IDs the model declares relevant."""
    import json

    candidates = [
        _make_ctx_item("m1", "Q: backpropagation? A: ..."),
        _make_ctx_item("m2", "Q: speed of light? A: ..."),
        _make_ctx_item("m3", "Q: learning rate? A: ..."),
    ]
    provider = FakeProvider(
        planner_outputs=[json.dumps({"relevant_ids": ["m1", "m3"]})]
    )
    result = filter_context_with_llm("explain backpropagation", candidates, provider)
    assert [item.id for item in result] == ["m1", "m3"]


def test_llm_context_filter_fallback_on_bad_json() -> None:
    """LLM context filter returns all candidates when the provider returns garbage."""
    candidates = [
        _make_ctx_item("m1", "some text"),
        _make_ctx_item("m2", "other text"),
    ]
    provider = FakeProvider(planner_outputs=["not json"])
    result = filter_context_with_llm("query", candidates, provider)
    assert [item.id for item in result] == ["m1", "m2"]


def test_compile_execute_and_analyze_v0() -> None:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    retriever = ConversationRetrieveTool()
    tools.register(retriever)
    provider = FakeProvider([conversation_summary_proposal()], model_text="A concise summary.")
    storage = InMemoryStorage()
    app = LLMASM(
        storage=storage,
        provider=provider,
        tool_registry=tools,
        runtime_config=RuntimeConfig(default_model="fake-model"),
        schema_registry=schemas,
    )
    workspace_id = app.create_workspace("test")

    answer = app.ask(workspace_id, "retrieve the conversation xyz and give me a summary of the content")

    assert answer == FinalAnswer(text="A concise summary.", sources=[])
    assert retriever.calls == 1
    run = next(iter(storage.runs.values()))
    assert run.status == RunStatus.SUCCEEDED
    analysis = app.query_run(run.id)
    assert len(analysis.tool_calls) == 1
    assert len(analysis.model_calls) == 1
    assert len(analysis.checkpoints) >= 2
    assert not analysis.failed_nodes()
    prompt = analysis.context_used_by_model_call(analysis.model_calls[0].id)
    assert "Project scope discussed" in str(prompt)
    viewer_graph = to_viewer_graph(analysis)
    assert len(viewer_graph["nodes"]) == 4
    assert len(viewer_graph["edges"]) == 3
    assert "flowchart TD" in to_mermaid(analysis)
    assert "digraph llmasm" in to_dot(analysis)


def test_compiler_canonicalizes_common_small_model_graph_variants() -> None:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    tools.register(ConversationRetrieveTool())
    provider = FakeProvider(
        [
            """
{
  "intent": "summarize conversation",
  "goal_action": "new",
  "nodes": [
    {
      "name": "intent",
      "kind": "intent",
      "metadata": {"output.text": "xyz"}
    },
    {
      "name": "conversation_store.retrieve",
      "kind": "tool",
      "metadata": {"input_schema": "RawText", "output_schema": "ConversationRecord"}
    },
    {
      "name": "summarize",
      "kind": "model",
      "metadata": {"input_schema": "ConversationText"},
      "output_schema": "Summary"
    },
    {
      "name": "final",
      "kind": "final"
    }
  ],
  "edges": [
    {"from_node": "intent", "to_node": "conversation_store.retrieve"},
    {
      "from_node": "conversation_store.retrieve",
      "to_node": "summarize",
      "transform": "extract_text"
    },
    {"from_node": "summarize", "to_node": "final"}
  ]
}
"""
        ],
        model_text="A concise summary.",
    )
    storage = InMemoryStorage()
    app = LLMASM(storage=storage, provider=provider, tool_registry=tools, schema_registry=schemas)
    workspace_id = app.create_workspace("test")

    task_graph_id = app.compile(workspace_id, "retrieve conversation xyz and summarize it")
    graph = storage.load_task_graph(task_graph_id)
    retrieve = next(node for node in graph.nodes if node.kind == NodeKind.TOOL)
    final = next(node for node in graph.nodes if node.kind == NodeKind.FINAL)

    assert retrieve.execution["tool"] == "conversation_store.retrieve"
    assert retrieve.input_schema == "RawText"
    assert retrieve.output_schema == "ConversationRecord"
    assert final.input_schema == "Summary"
    assert final.output_schema == "FinalAnswer"


def test_compiler_synthesizes_simple_qa_chain_edges() -> None:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    tools.register(RawSearchTool())
    provider = FakeProvider(
        [
            """
{
  "intent": "summarize conversation",
  "goal_action": "new",
  "nodes": [
    {
      "name": "intent",
      "kind": "intent",
      "output_schema": "RawText",
      "metadata": {"output.text": "xyz"}
    },
    {
      "name": "raw.search",
      "kind": "tool"
    },
    {
      "name": "summarize",
      "kind": "model",
      "output_schema": "Summary"
    },
    {
      "name": "final",
      "kind": "final"
    }
  ],
  "edges": []
}
"""
        ],
        model_text="A concise summary.",
    )
    storage = InMemoryStorage()
    app = LLMASM(storage=storage, provider=provider, tool_registry=tools, schema_registry=schemas)
    workspace_id = app.create_workspace("test")

    task_graph_id = app.compile(workspace_id, "search raw text and summarize it")
    graph = storage.load_task_graph(task_graph_id)

    assert len(graph.task_edges) == 3


def test_compiler_repair_persists_failure() -> None:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    tools.register(ConversationRetrieveTool())
    provider = FakeProvider(["not-json"])
    storage = InMemoryStorage()
    app = LLMASM(storage=storage, provider=provider, tool_registry=tools, schema_registry=schemas)
    workspace_id = app.create_workspace("test")
    with pytest.raises(Exception):
        app.compile(workspace_id, "retrieve conversation xyz")
    assert storage.compilation_failures


def test_tool_cache_reuses_artifact_without_invocation() -> None:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    retriever = ConversationRetrieveTool()
    tools.register(retriever)
    provider = FakeProvider([conversation_summary_proposal()], model_text="Summary.")
    storage = InMemoryStorage()
    app = LLMASM(storage=storage, provider=provider, tool_registry=tools, schema_registry=schemas)
    workspace_id = app.create_workspace("test")
    task_graph_id = app.compile(workspace_id, "retrieve conversation xyz")

    app.run(task_graph_id)
    app.run(task_graph_id)

    assert retriever.calls == 1


def test_embeddings_lifecycle() -> None:
    provider = FakeProvider()
    config = RuntimeConfig(embeddings_enabled=False)
    store = NullEmbeddingStore()
    assert embed_and_persist("hello", "prompt", "prompt_1", config, provider, store) is None
    assert provider.embed_calls == 0

    memory_store = InMemoryEmbeddingStore()
    config.embeddings_enabled = True
    ref = embed_and_persist("hello", "prompt", "prompt_1", config, provider, memory_store)
    assert ref is not None
    again = embed_and_persist("hello", "prompt", "prompt_1", config, provider, memory_store)
    assert again == ref
    assert provider.embed_calls == 1


def test_embedding_store_cosine_ranking() -> None:
    store = InMemoryEmbeddingStore()
    item_a = ConversationRecord(id="a", text="alpha", metadata={})
    item_b = ConversationRecord(id="b", text="beta", metadata={})
    from llmasm.graph.models import Artifact, EmbeddingRef

    artifact_a = Artifact(id="artifact_a", run_id="run_1", node_id="node_1", port="output", content_json=item_a.model_dump())
    artifact_b = Artifact(id="artifact_b", run_id="run_1", node_id="node_1", port="output", content_json=item_b.model_dump())
    store.attach_item("artifact", artifact_a.id, artifact_a)
    store.attach_item("artifact", artifact_b.id, artifact_b)
    store.persist(EmbeddingRef(id="memory_a", owner_type="artifact", owner_id=artifact_a.id, model="m", dimensions=2, text_hash="a"), [1.0, 0.0])
    store.persist(EmbeddingRef(id="memory_b", owner_type="artifact", owner_id=artifact_b.id, model="m", dimensions=2, text_hash="b"), [0.0, 1.0])
    matches = store.search_similar([0.9, 0.1], {}, 2)
    assert matches[0].item.id == "artifact_a"


def test_unsupported_kind_fails_with_kind_name() -> None:
    schemas = default_schema_registry()
    storage = InMemoryStorage()
    workspace = WorkspaceGraph(id="workspace_1", name="test")
    storage.create_workspace_graph(workspace)
    from llmasm.graph.models import TaskGraph

    node = Node(
        id="node_1",
        workspace_graph_id=workspace.id,
        task_graph_id="taskgraph_1",
        kind=NodeKind.MEMORY_QUERY,
        name="mem",
    )
    storage.persist_task_graph(TaskGraph(id="taskgraph_1", workspace_graph_id=workspace.id, nodes=[node]))
    run = Run(id="run_1", workspace_graph_id=workspace.id, task_graph_id="taskgraph_1")
    storage.create_run(run)
    executor = Executor(
        storage=storage,
        tool_registry=ToolRegistry(schemas),
        provider=FakeProvider(),
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(),
    )
    with pytest.raises(Exception, match="memory_query"):
        executor.execute(run.id)


def test_executor_accepts_planner_intent_text_variants() -> None:
    schemas = default_schema_registry()
    executor = Executor(
        storage=InMemoryStorage(),
        tool_registry=ToolRegistry(schemas),
        provider=FakeProvider(),
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(),
    )

    assert executor._intent_raw_text("text", {"text": "question text"}) == "question text"
    assert executor._intent_raw_text({"output.text": "question text"}, {}) == "question text"


def test_select_context_vector_path_returns_memory_items() -> None:
    """select_context should embed the query and surface matching MemoryItems."""
    from llmasm.graph.models import EmbeddingRef, MemoryItem, Run
    from llmasm.runtime.context import select_context
    from llmasm.storage.embeddings import InMemoryEmbeddingStore

    storage = InMemoryStorage()
    workspace_id = new_id("workspace")
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))

    item = MemoryItem(
        id=new_id("memory"),
        workspace_graph_id=workspace_id,
        kind="fact",
        text="The sky is blue.",
    )
    storage.persist_memory_item(item)

    emb_store = InMemoryEmbeddingStore()
    emb_store.attach_item("memory_item", item.id, item)
    emb_store.persist(
        EmbeddingRef(
            id=new_id("memory"),
            owner_type="memory_item",
            owner_id=item.id,
            model="nomic-embed-text",
            dimensions=2,
            text_hash="abc",
        ),
        [1.0, 0.0],  # fixed vector; FakeProvider will return a compatible direction
    )

    tg_id = new_id("taskgraph")
    node = Node(
        id=new_id("node"),
        workspace_graph_id=workspace_id,
        task_graph_id=tg_id,
        kind=NodeKind.MODEL,
        name="summarize",
        metadata={"instruction": "summarize"},
    )
    storage.persist_task_graph(
        TaskGraph(id=tg_id, workspace_graph_id=workspace_id, nodes=[node])
    )
    run = Run(id=new_id("run"), workspace_graph_id=workspace_id, task_graph_id=tg_id)
    storage.create_run(run)

    provider = FakeProvider()
    config = RuntimeConfig(embeddings_enabled=True, embedding_model="nomic-embed-text")

    result = select_context(
        storage=storage,
        runtime_config=config,
        run=run,
        node=node,
        direct_inputs={},
        embedding_store=emb_store,
        provider=provider,
    )

    assert provider.embed_calls == 1
    assert any(ci.id == item.id for ci in result.items)
    matched = next(ci for ci in result.items if ci.id == item.id)
    assert matched.score > 0
    assert matched.text == item.text


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

def _build_router_graph(workspace_id: str, storage: InMemoryStorage, *, not_found_input: bool = False) -> Run:
    """Build intent → router → {found: model → final_ok} / {missing: final_missing} graph."""
    from llmasm.graph.models import TaskEdge, TaskGraph
    from llmasm.schemas import NotFound, RawText

    tg_id = new_id("taskgraph")

    n_intent = Node(id=new_id("node"), workspace_graph_id=workspace_id, task_graph_id=tg_id, kind=NodeKind.INTENT, name="intent", output_schema="RawText", metadata={"output": {"text": "test"}})
    n_router = Node(id=new_id("node"), workspace_graph_id=workspace_id, task_graph_id=tg_id, kind=NodeKind.ROUTER, name="router", input_schema="RawText", output_schema="RoutingDecision", execution={"mode": "deterministic", "default_branch": "found", "not_found_branch": "missing"})
    n_model = Node(id=new_id("node"), workspace_graph_id=workspace_id, task_graph_id=tg_id, kind=NodeKind.MODEL, name="summarise", input_schema="RawText", output_schema="Summary", execution={"model": "fake-model"}, metadata={"instruction": "summarise"})
    n_final_ok = Node(id=new_id("node"), workspace_graph_id=workspace_id, task_graph_id=tg_id, kind=NodeKind.FINAL, name="final_ok", input_schema="Summary", output_schema="FinalAnswer")
    n_final_missing = Node(id=new_id("node"), workspace_graph_id=workspace_id, task_graph_id=tg_id, kind=NodeKind.FINAL, name="final_missing", input_schema="RawText", output_schema="FinalAnswer")

    if not_found_input:
        n_intent = n_intent.model_copy(update={"output_schema": "NotFound", "metadata": {"output": {"resource_type": "doc", "resource_id": "x", "detail": "not found"}}})
        n_router = n_router.model_copy(update={"input_schema": "NotFound"})

    edges = [
        TaskEdge(id=new_id("edge"), workspace_graph_id=workspace_id, task_graph_id=tg_id, from_node_id=n_intent.id, from_port="output", to_node_id=n_router.id, to_port="input"),
        TaskEdge(id=new_id("edge"), workspace_graph_id=workspace_id, task_graph_id=tg_id, from_node_id=n_router.id, from_port="output", to_node_id=n_model.id, to_port="input", metadata={"branch": "found"}),
        TaskEdge(id=new_id("edge"), workspace_graph_id=workspace_id, task_graph_id=tg_id, from_node_id=n_router.id, from_port="output", to_node_id=n_final_missing.id, to_port="input", metadata={"branch": "missing"}),
        TaskEdge(id=new_id("edge"), workspace_graph_id=workspace_id, task_graph_id=tg_id, from_node_id=n_model.id, from_port="output", to_node_id=n_final_ok.id, to_port="input"),
    ]
    graph = TaskGraph(id=tg_id, workspace_graph_id=workspace_id, nodes=[n_intent, n_router, n_model, n_final_ok, n_final_missing], task_edges=edges)
    storage.persist_task_graph(graph)
    run = Run(id=new_id("run"), workspace_graph_id=workspace_id, task_graph_id=tg_id)
    storage.create_run(run)
    return run, graph


def test_router_deterministic_selects_found_branch() -> None:
    from llmasm.graph.models import NodeStatus

    schemas = default_schema_registry()
    storage = InMemoryStorage()
    workspace_id = new_id("workspace")
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))
    run, graph = _build_router_graph(workspace_id, storage, not_found_input=False)

    executor = Executor(
        storage=storage,
        tool_registry=ToolRegistry(schemas),
        provider=FakeProvider(model_text="summary text"),
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(default_model="fake-model"),
    )
    completed_run = executor.execute(run.id)

    assert completed_run.status == RunStatus.SUCCEEDED
    states = {s.node_id: s for s in storage.list_run_node_states(run.id)}
    nodes_by_name = {n.name: n for n in graph.nodes}

    assert states[nodes_by_name["summarise"].id].status == NodeStatus.SUCCEEDED
    assert states[nodes_by_name["final_ok"].id].status == NodeStatus.SUCCEEDED
    assert states[nodes_by_name["final_missing"].id].status == NodeStatus.SKIPPED


def test_router_deterministic_selects_not_found_branch() -> None:
    from llmasm.graph.models import NodeStatus

    schemas = default_schema_registry()
    storage = InMemoryStorage()
    workspace_id = new_id("workspace")
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))
    run, graph = _build_router_graph(workspace_id, storage, not_found_input=True)

    executor = Executor(
        storage=storage,
        tool_registry=ToolRegistry(schemas),
        provider=FakeProvider(model_text="summary text"),
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(default_model="fake-model"),
    )
    completed_run = executor.execute(run.id)

    assert completed_run.status == RunStatus.SUCCEEDED
    states = {s.node_id: s for s in storage.list_run_node_states(run.id)}
    nodes_by_name = {n.name: n for n in graph.nodes}

    assert states[nodes_by_name["summarise"].id].status == NodeStatus.SKIPPED
    assert states[nodes_by_name["final_ok"].id].status == NodeStatus.SKIPPED
    assert states[nodes_by_name["final_missing"].id].status == NodeStatus.SUCCEEDED


def test_router_model_assisted_selects_branch() -> None:
    from llmasm.graph.models import NodeStatus, TaskEdge, TaskGraph

    schemas = default_schema_registry()
    storage = InMemoryStorage()
    workspace_id = new_id("workspace")
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))

    tg_id = new_id("taskgraph")
    n_intent = Node(id=new_id("node"), workspace_graph_id=workspace_id, task_graph_id=tg_id, kind=NodeKind.INTENT, name="intent", output_schema="RawText", metadata={"output": {"text": "test"}})
    n_router = Node(id=new_id("node"), workspace_graph_id=workspace_id, task_graph_id=tg_id, kind=NodeKind.ROUTER, name="router", input_schema="RawText", output_schema="RoutingDecision", execution={"mode": "model", "model": "fake-model", "branches": ["yes", "no"]}, metadata={"instruction": "pick yes or no"})
    n_yes = Node(id=new_id("node"), workspace_graph_id=workspace_id, task_graph_id=tg_id, kind=NodeKind.FINAL, name="final_yes", input_schema="RawText", output_schema="FinalAnswer")
    n_no = Node(id=new_id("node"), workspace_graph_id=workspace_id, task_graph_id=tg_id, kind=NodeKind.FINAL, name="final_no", input_schema="RawText", output_schema="FinalAnswer")
    edges = [
        TaskEdge(id=new_id("edge"), workspace_graph_id=workspace_id, task_graph_id=tg_id, from_node_id=n_intent.id, from_port="output", to_node_id=n_router.id, to_port="input"),
        TaskEdge(id=new_id("edge"), workspace_graph_id=workspace_id, task_graph_id=tg_id, from_node_id=n_router.id, from_port="output", to_node_id=n_yes.id, to_port="input", metadata={"branch": "yes"}),
        TaskEdge(id=new_id("edge"), workspace_graph_id=workspace_id, task_graph_id=tg_id, from_node_id=n_router.id, from_port="output", to_node_id=n_no.id, to_port="input", metadata={"branch": "no"}),
    ]
    graph = TaskGraph(id=tg_id, workspace_graph_id=workspace_id, nodes=[n_intent, n_router, n_yes, n_no], task_edges=edges)
    storage.persist_task_graph(graph)
    run = Run(id=new_id("run"), workspace_graph_id=workspace_id, task_graph_id=tg_id)
    storage.create_run(run)

    # FakeProvider returns this JSON for the model-assisted router call
    provider = FakeProvider(model_text='{"selected_branch": "yes", "reason": "test"}')
    executor = Executor(
        storage=storage,
        tool_registry=ToolRegistry(schemas),
        provider=provider,
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(default_model="fake-model"),
    )
    completed_run = executor.execute(run.id)

    assert completed_run.status == RunStatus.SUCCEEDED
    states = {s.node_id: s for s in storage.list_run_node_states(run.id)}
    assert states[n_yes.id].status == NodeStatus.SUCCEEDED
    assert states[n_no.id].status == NodeStatus.SKIPPED
