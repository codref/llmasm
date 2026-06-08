"""Tests for deterministic chat fast path."""

from __future__ import annotations

import json

import pytest

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
from llmasm.schemas import FinalAnswer
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
