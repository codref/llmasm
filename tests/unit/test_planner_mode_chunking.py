"""Tests for planner-mode chunking and summary nodes."""

from __future__ import annotations

import json
from typing import Any

from llmasm.api import LLMASM
from llmasm.config import RuntimeConfig
from llmasm.conversation.ingestion import SOURCE_PLACEHOLDER
from llmasm.conversation.memory import (
    get_source_passages,
    list_conversation_memory,
    store_assistant_answer,
    store_user_question,
)
from llmasm.graph.models import MemoryItem, Node, NodeKind, Run, WorkspaceGraph
from llmasm.graph.registry import default_schema_registry
from llmasm.graph.transforms import default_transform_registry
from llmasm.runtime.context import select_context
from llmasm.storage.embeddings import InMemoryEmbeddingStore, NullEmbeddingStore
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.registry import ToolRegistry
from tests.unit.fakes import FakeProvider


def _make_long_source_text(sentences: int = 30) -> str:
    return " ".join(f"Sentence number {i} contains some words about the topic." for i in range(1, sentences + 1))


def _simple_model_text(prompt: str) -> str:
    try:
        parsed = json.loads(prompt)
        instruction = parsed.get("instruction", "")
    except (json.JSONDecodeError, AttributeError):
        instruction = prompt
    if "summarize" in instruction.lower():
        return json.dumps({"text": "A concise summary."})
    return json.dumps({"text": "answer text", "sources": []})


def _make_chunking_app(
    *,
    planner_outputs: list[str] | None = None,
    model_text: Any = _simple_model_text,
    embeddings_enabled: bool = False,
) -> tuple[LLMASM, InMemoryStorage, FakeProvider]:
    storage = InMemoryStorage()
    provider = FakeProvider(planner_outputs=planner_outputs, model_text=model_text)
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    embedding_store = InMemoryEmbeddingStore() if embeddings_enabled else NullEmbeddingStore()
    app = LLMASM(
        storage=storage,
        provider=provider,
        tool_registry=tools,
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(
            default_model="fake-model",
            embeddings_enabled=embeddings_enabled,
            chunking_enabled=True,
            chunking_trigger_tokens=20,
            chunk_target_tokens=10,
            chunk_overlap_tokens=2,
            chunking_summary_enabled=True,
        ),
        embedding_store=embedding_store,
    )
    return app, storage, provider


def _simple_proposal() -> str:
    return json.dumps(
        {
            "intent": "answer question",
            "goal_action": "continue",
            "nodes": [
                {
                    "name": "intent",
                    "kind": "intent",
                    "output_schema": "RawText",
                    "execution": {"options": {}, "allow_cache": False},
                    "metadata": {"output": {"text": "placeholder"}},
                },
                {
                    "name": "answer",
                    "kind": "model",
                    "input_schema": "RawText",
                    "output_schema": "Summary",
                    "execution": {"provider": "fake", "model": "fake-model", "allow_cache": False},
                    "metadata": {"instruction": "Answer the question."},
                },
                {
                    "name": "final",
                    "kind": "final",
                    "input_schema": "Summary",
                    "output_schema": "FinalAnswer",
                },
            ],
            "edges": [
                {"from_node": "intent", "from_port": "output", "to_node": "answer", "to_port": "input"},
                {"from_node": "answer", "from_port": "output", "to_node": "final", "to_port": "input"},
            ],
        }
    )


def _summary_proposal() -> str:
    return json.dumps(
        {
            "intent": "summarize and answer",
            "goal_action": "continue",
            "nodes": [
                {
                    "name": "intent",
                    "kind": "intent",
                    "output_schema": "RawText",
                    "execution": {"options": {}, "allow_cache": False},
                    "metadata": {"output": {"text": SOURCE_PLACEHOLDER}},
                },
                {
                    "name": "summarize_source",
                    "kind": "model",
                    "input_schema": "RawText",
                    "output_schema": "Summary",
                    "execution": {"provider": "fake", "model": "fake-model", "allow_cache": False},
                    "metadata": {
                        "instruction": "Summarize the stored source.",
                        "is_summary_node": True,
                    },
                },
                {
                    "name": "answer",
                    "kind": "model",
                    "input_schema": "Summary",
                    "output_schema": "Summary",
                    "execution": {"provider": "fake", "model": "fake-model", "allow_cache": False},
                    "metadata": {"instruction": "Answer based on the summary."},
                },
                {
                    "name": "final",
                    "kind": "final",
                    "input_schema": "Summary",
                    "output_schema": "FinalAnswer",
                },
            ],
            "edges": [
                {"from_node": "intent", "from_port": "output", "to_node": "summarize_source", "to_port": "input"},
                {
                    "from_node": "summarize_source",
                    "from_port": "output",
                    "to_node": "answer",
                    "to_port": "input",
                },
                {"from_node": "answer", "from_port": "output", "to_node": "final", "to_port": "input"},
            ],
        }
    )


class TestAskChunksLongSource:
    def test_long_source_is_chunked_and_planner_sees_placeholder(self) -> None:
        app, storage, provider = _make_chunking_app(planner_outputs=[_simple_proposal()])
        workspace_id = app.create_workspace("test")
        source = _make_long_source_text()
        answer = app.ask(workspace_id, source)

        assert answer.text == "answer text"

        items = get_source_passages(storage, workspace_id)
        chunks = [item for item in items if not item.metadata.get("is_summary")]
        assert len(chunks) > 1
        assert all(item.metadata.get("source_id") for item in chunks)

        planner_calls = [p for p in provider.generate_prompts if "TaskGraphProposal" in p]
        assert len(planner_calls) == 1
        assert SOURCE_PLACEHOLDER in planner_calls[0]
        assert source not in planner_calls[0]

    def test_short_prompt_is_unchanged(self) -> None:
        app, storage, provider = _make_chunking_app(planner_outputs=[_simple_proposal()])
        workspace_id = app.create_workspace("test")
        answer = app.ask(workspace_id, "What is the capital of France?")

        assert answer.text == "answer text"
        assert get_source_passages(storage, workspace_id) == []

        planner_calls = [p for p in provider.generate_prompts if "TaskGraphProposal" in p]
        assert "What is the capital of France?" in planner_calls[0]


class TestPlannerEmitsSummaryNode:
    def test_summary_node_is_accepted_and_normalized(self) -> None:
        app, storage, _provider = _make_chunking_app(planner_outputs=[_summary_proposal()])
        workspace_id = app.create_workspace("test")
        app.ask(workspace_id, _make_long_source_text())

        graphs = list(storage.task_graphs.values())
        assert len(graphs) == 1
        summary_nodes = [n for n in graphs[0].nodes if n.name == "summarize_source"]
        assert len(summary_nodes) == 1
        assert summary_nodes[0].kind == NodeKind.MODEL
        assert summary_nodes[0].output_schema == "Summary"
        assert summary_nodes[0].metadata.get("is_summary_node") is True


class TestSummaryArtifactStoredAfterRun:
    def test_summary_memory_item_is_created(self) -> None:
        app, storage, _provider = _make_chunking_app(planner_outputs=[_summary_proposal()])
        workspace_id = app.create_workspace("test")
        app.ask(workspace_id, _make_long_source_text())

        items = get_source_passages(storage, workspace_id)
        summaries = [item for item in items if item.metadata.get("is_summary")]
        assert len(summaries) == 1
        assert summaries[0].text == "A concise summary."
        assert summaries[0].metadata.get("source_id")

        chunks = [item for item in items if not item.metadata.get("is_summary")]
        assert len(chunks) > 0
        assert any(item.metadata.get("source_id") == summaries[0].metadata.get("source_id") for item in chunks)


class TestSelectContextPrefersSummary:
    def test_summary_placed_before_chunks(self) -> None:
        storage = InMemoryStorage()
        provider = FakeProvider()
        runtime_config = RuntimeConfig(
            default_model="fake-model",
            embeddings_enabled=False,
            default_context_tokens=200,
        )
        workspace_id = "workspace_test"
        storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))

        def make_memory(kind: str, text: str, **metadata: Any) -> MemoryItem:
            return MemoryItem(
                id=f"memory_{kind}_{text[:10]}",
                workspace_graph_id=workspace_id,
                kind=kind,
                text=text,
                metadata=dict(metadata),
            )

        summary = make_memory("source_passage", "This is the answer summary.", is_summary=True)
        chunk = make_memory("source_passage", "This is a detailed chunk with extra question words.", is_summary=False)
        question = make_memory("user_question", "What is the answer about?")

        for item in [chunk, question, summary]:
            storage.persist_memory_item(item)

        run = Run(id="run_1", workspace_graph_id=workspace_id, task_graph_id="tg_1")
        node = Node(
            id="node_1",
            workspace_graph_id=workspace_id,
            task_graph_id="tg_1",
            kind=NodeKind.MODEL,
            name="answer",
            input_schema="RawText",
            output_schema="Summary",
            metadata={"instruction": "Answer the question."},
        )

        # Mock retrieve_workspace_context so it returns our items via word overlap.
        selected = select_context(
            storage=storage,
            runtime_config=runtime_config,
            run=run,
            node=node,
            direct_inputs={},
            embedding_store=NullEmbeddingStore(),
            provider=provider,
        )

        item_texts = [item.text for item in selected.items]
        assert "This is the answer summary." in item_texts
        summary_index = item_texts.index("This is the answer summary.")
        chunk_index = item_texts.index("This is a detailed chunk with extra question words.")
        assert summary_index < chunk_index


class TestNoMonolithicTurnMemory:
    def test_ask_plus_structured_storage_creates_no_turn_items(self) -> None:
        app, storage, _provider = _make_chunking_app(planner_outputs=[_simple_proposal()])
        workspace_id = app.create_workspace("test")
        turn_info: dict[str, Any] = {}
        answer = app.ask(workspace_id, "What is the capital?", out_info=turn_info)

        store_user_question(
            workspace_id,
            "What is the capital?",
            storage=storage,
            runtime_config=app.runtime_config,
            provider=app.provider,
            embedding_store=app.embedding_store or NullEmbeddingStore(),
            source_run_id=turn_info.get("run_id"),
            turn=0,
        )
        store_assistant_answer(
            workspace_id,
            answer.text,
            storage=storage,
            runtime_config=app.runtime_config,
            provider=app.provider,
            embedding_store=app.embedding_store or NullEmbeddingStore(),
            source_run_id=turn_info.get("run_id"),
            turn=0,
        )

        memory = list_conversation_memory(storage, workspace_id)
        turn_items = [item for item in memory if item.kind == "turn"]
        assert turn_items == []
        assert any(item.kind == "user_question" for item in memory)
        assert any(item.kind == "assistant_answer" for item in memory)
