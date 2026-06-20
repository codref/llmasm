"""Tests for deterministic chat fast path."""

from __future__ import annotations

import json


from llmasm.api import LLMASM
from llmasm.config import RuntimeConfig
from llmasm.conversation.memory import (
    get_recent_user_questions,
    get_source_passages,
    list_conversation_memory,
)
from llmasm.graph.models import NodeKind
from llmasm.graph.registry import default_schema_registry
from llmasm.graph.transforms import default_transform_registry
from llmasm.storage.embeddings import NullEmbeddingStore
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.registry import ToolRegistry
from tests.unit.fakes import FakeProvider


def _chat_model_text(prompt: str) -> str:
    """Fake model that returns answers based on prompt content."""
    # The fast-path prompt is a JSON object containing an "instruction" field.
    try:
        parsed = json.loads(prompt)
        instruction = parsed.get("instruction", "")
    except (json.JSONDecodeError, AttributeError):
        instruction = prompt

    # Extract current question from the instruction
    question = ""
    for line in instruction.splitlines():
        if line.startswith("Current question:"):
            question = line.split(":", 1)[1].strip().lower()
            break
    if not question:
        question = instruction.lower()

    if "burroughs" in question or "boroughs" in question:
        return json.dumps({"text": "There are five boroughs.", "sources": []})
    if "what city" in question or "in what city" in question:
        return json.dumps({"text": "New York City.", "sources": []})
    if "and state" in question:
        return json.dumps({"text": "New York.", "sources": []})
    return json.dumps({"text": "generic answer", "sources": []})


def _make_app() -> tuple[LLMASM, InMemoryStorage, FakeProvider]:
    storage = InMemoryStorage()
    provider = FakeProvider(model_text=_chat_model_text)
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    app = LLMASM(
        storage=storage,
        provider=provider,
        tool_registry=tools,
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(
            default_model="fake-model",
            embeddings_enabled=False,
        ),
        embedding_store=NullEmbeddingStore(),
    )
    return app, storage, provider


class TestConversationFastPath:
    def test_setup_instruction_acknowledged(self) -> None:
        app, storage, _provider = _make_app()
        workspace_id = app.create_workspace("test")
        answer = app.chat(
            workspace_id,
            "I will ask you several questions about the following passage. Please read it carefully.",
        )
        assert "understood" in answer.text.lower() or "keep that in mind" in answer.text.lower()
        memory = list_conversation_memory(storage, workspace_id)
        notes = [m for m in memory if m.kind == "system_note"]
        assert len(notes) == 1

    def test_source_passage_stored(self) -> None:
        app, storage, _provider = _make_app()
        workspace_id = app.create_workspace("test")
        passage = "Staten Island is one of the five boroughs of New York City in New York."
        answer = app.chat(workspace_id, passage)
        assert "got it" in answer.text.lower() or "saved" in answer.text.lower()
        sources = get_source_passages(storage, workspace_id)
        assert len(sources) == 1
        assert "five boroughs" in sources[0].text

    def test_question_answered_with_source(self) -> None:
        app, storage, provider = _make_app()
        workspace_id = app.create_workspace("test")
        # Prime the workspace with a source passage
        app.chat(
            workspace_id,
            "Staten Island is one of the five boroughs of New York City in New York.",
        )
        answer = app.chat(workspace_id, "How many burroughs are there?")
        assert "five" in answer.text.lower()
        # Verify memory
        questions = get_recent_user_questions(storage, workspace_id)
        assert questions == ["How many burroughs are there?"]
        # Verify no planner calls (format_schema should be None for model calls)
        planner_calls = [p for p in provider.generate_prompts if "TaskGraphProposal" in p]
        assert len(planner_calls) == 0

    def test_followup_question(self) -> None:
        app, storage, _provider = _make_app()
        workspace_id = app.create_workspace("test")
        app.chat(
            workspace_id,
            "Staten Island is one of the five boroughs of New York City in New York.",
        )
        app.chat(workspace_id, "How many burroughs are there?")
        answer = app.chat(workspace_id, "in what city?")
        assert "new york city" in answer.text.lower()
        questions = get_recent_user_questions(storage, workspace_id)
        assert "in what city?" in questions

    def test_and_state_followup(self) -> None:
        app, storage, _provider = _make_app()
        workspace_id = app.create_workspace("test")
        app.chat(
            workspace_id,
            "Staten Island is one of the five boroughs of New York City in New York.",
        )
        app.chat(workspace_id, "How many burroughs are there?")
        app.chat(workspace_id, "in what city?")
        answer = app.chat(workspace_id, "and state?")
        assert "new york" in answer.text.lower()

    def test_fast_path_graph_shape(self) -> None:
        app, storage, _provider = _make_app()
        workspace_id = app.create_workspace("test")
        app.chat(
            workspace_id,
            "Staten Island is one of the five boroughs of New York City in New York.",
        )
        app.chat(workspace_id, "How many burroughs are there?")
        # Load the most recent task graph
        graphs = list(storage.task_graphs.values())
        chat_graphs = [g for g in graphs if g.metadata.get("fast_path")]
        assert len(chat_graphs) >= 1
        graph = chat_graphs[-1]
        kinds = {node.kind for node in graph.nodes}
        assert kinds == {NodeKind.INTENT, NodeKind.MODEL, NodeKind.FINAL}

    def test_strict_grounding_in_prompt(self) -> None:
        app, _storage, provider = _make_app()
        workspace_id = app.create_workspace("test")
        app.chat(
            workspace_id,
            "Staten Island is one of the five boroughs of New York City in New York.",
        )
        provider.generate_prompts.clear()
        app.chat(workspace_id, "How many burroughs are there?")
        model_prompt = provider.generate_prompts[-1]
        assert "Answer using ONLY" in model_prompt
        assert "imply or contradict" in model_prompt
        assert "you may use your own knowledge" not in model_prompt

    def test_no_embeddings_called_when_disabled(self) -> None:
        app, _storage, provider = _make_app()
        workspace_id = app.create_workspace("test")
        app.chat(
            workspace_id,
            "Staten Island is one of the five boroughs of New York City in New York.",
        )
        assert provider.embed_calls == 0
        app.chat(workspace_id, "How many burroughs are there?")
        assert provider.embed_calls == 0

    def test_existing_ask_still_uses_planner(self) -> None:
        from tests.unit.fakes import ConversationRetrieveTool, conversation_summary_proposal

        storage = InMemoryStorage()
        provider = FakeProvider(
            planner_outputs=[conversation_summary_proposal()],
            model_text=json.dumps({"text": "summary text", "sources": []}),
        )
        schemas = default_schema_registry()
        tools = ToolRegistry(schemas)
        tools.register(ConversationRetrieveTool())
        app = LLMASM(
            storage=storage,
            provider=provider,
            tool_registry=tools,
            schema_registry=schemas,
            transform_registry=default_transform_registry(),
            runtime_config=RuntimeConfig(
                default_model="fake-model",
                embeddings_enabled=False,
            ),
            embedding_store=NullEmbeddingStore(),
        )
        workspace_id = app.create_workspace("test")
        answer = app.ask(workspace_id, "retrieve conversation xyz and summarize")
        assert answer.text == "summary text"
        # Verify planner WAS called (format_schema is not None)
        planner_calls = [p for p in provider.generate_prompts if "TaskGraphProposal" in p]
        assert len(planner_calls) > 0


def _chunking_model_text(prompt: str) -> str:
    try:
        parsed = json.loads(prompt)
        instruction = parsed.get("instruction", "")
    except (json.JSONDecodeError, AttributeError):
        instruction = prompt

    if "summarize" in instruction.lower() and "source text" in instruction.lower():
        return json.dumps({"text": "A concise summary of the source."})

    question = ""
    for line in instruction.splitlines():
        if line.startswith("Current question:"):
            question = line.split(":", 1)[1].strip().lower()
            break
    if "how many" in question:
        return json.dumps({"text": "Five.", "sources": []})
    return json.dumps({"text": "generic answer", "sources": []})


def _make_chunking_app(
    *,
    chunking_enabled: bool = True,
    trigger_tokens: int = 10,
) -> tuple[LLMASM, InMemoryStorage, FakeProvider]:
    storage = InMemoryStorage()
    provider = FakeProvider(model_text=_chunking_model_text)
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    app = LLMASM(
        storage=storage,
        provider=provider,
        tool_registry=tools,
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(
            default_model="fake-model",
            embeddings_enabled=False,
            chunking_enabled=chunking_enabled,
            chunking_trigger_tokens=trigger_tokens,
            chunk_target_tokens=8,
            chunk_overlap_tokens=2,
            chunking_summary_enabled=True,
        ),
        embedding_store=NullEmbeddingStore(),
    )
    return app, storage, provider


class TestChunkingFastPath:
    def test_long_source_passage_is_chunked_and_summarized(self) -> None:
        app, storage, _provider = _make_chunking_app()
        workspace_id = app.create_workspace("test")
        passage = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. Eleven. Twelve."
        answer = app.chat(workspace_id, passage)
        assert "got it" in answer.text.lower() or "saved" in answer.text.lower()

        items = get_source_passages(storage, workspace_id)
        chunks = [item for item in items if not item.metadata.get("is_summary")]
        summaries = [item for item in items if item.metadata.get("is_summary")]

        assert len(chunks) > 1
        assert all(item.kind == "source_passage" for item in chunks)
        assert all(item.metadata.get("source_id") for item in chunks)
        assert len(summaries) == 1
        assert summaries[0].text == "A concise summary of the source."
        assert summaries[0].metadata.get("source_id")

    def test_chunking_disabled_stores_single_passage(self) -> None:
        app, storage, _provider = _make_chunking_app(chunking_enabled=False, trigger_tokens=5)
        workspace_id = app.create_workspace("test")
        passage = "This is a long enough source passage to exceed the trigger. " * 3
        answer = app.chat(workspace_id, passage)
        assert "got it" in answer.text.lower() or "saved" in answer.text.lower()

        sources = get_source_passages(storage, workspace_id)
        assert len(sources) == 1
        assert not sources[0].metadata.get("is_summary")

    def test_summary_node_graph_shape(self) -> None:
        app, storage, _provider = _make_chunking_app()
        workspace_id = app.create_workspace("test")
        app.chat(workspace_id, "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. Eleven. Twelve.")

        summary_graphs = [
            g for g in storage.task_graphs.values()
            if g.metadata.get("summary_graph")
        ]
        assert len(summary_graphs) == 1
        kinds = {node.kind for node in summary_graphs[0].nodes}
        assert kinds == {NodeKind.INTENT, NodeKind.MODEL, NodeKind.FINAL}
        summary_nodes = [n for n in summary_graphs[0].nodes if n.name == "summary"]
        assert len(summary_nodes) == 1
        assert summary_nodes[0].output_schema == "Summary"
