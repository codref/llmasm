"""Tests for conversation retrieval (RAG) components."""

from __future__ import annotations

from llmasm.config import RuntimeConfig
from llmasm.conversation.retrieval import (
    LLMRewritePreparer,
    PassthroughPreparer,
    compose_instruction,
    retrieve_context,
)
from llmasm.graph.registry import default_schema_registry
from llmasm.graph.transforms import default_transform_registry
from llmasm.schemas import FinalAnswer
from llmasm.storage.embeddings import InMemoryEmbeddingStore, NullEmbeddingStore
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.registry import ToolRegistry
from tests.unit.fakes import FakeProvider


def test_passthrough_preparer() -> None:
    p = PassthroughPreparer()
    assert p.prepare("Which one?", [], FakeProvider(), "fake-model") == "Which one?"


def test_llm_rewrite_preparer() -> None:
    provider = FakeProvider(model_text="Which spinoff did Dennis Farina reprise his role on?")
    p = LLMRewritePreparer()
    result = p.prepare(
        "Which one?",
        [("What happened in 2004?", "He joined Law & Order playing Detective Joe Fontana.")],
        provider,
        "fake-model",
    )
    assert "Which spinoff did Dennis Farina reprise his role on?" == result
    # Verify the provider was called with a rewrite prompt
    assert "replace pronouns" in provider.generate_prompts[0]


def test_llm_rewrite_preparer_fallback_on_short_output() -> None:
    provider = FakeProvider(model_text="no")
    p = LLMRewritePreparer()
    result = p.prepare(
        "Which one?",
        [("What happened in 2004?", "He joined Law & Order.")],
        provider,
        "fake-model",
    )
    # Falls back to original prompt because rewrite is shorter than original
    assert result == "Which one?"


def test_compose_instruction_with_passages_and_qa() -> None:
    instruction = compose_instruction(
        prompt="How long?",
        dialogue_type="followup_question",
        source_passages=["He was on the show for two years."],
        qa_pairs=[("Was he on for five years?", "No, he was on for two years.")],
    )
    assert "Answer using ONLY the source passages below" in instruction
    assert "He was on the show for two years" in instruction
    assert "Q: Was he on for five years?" in instruction
    assert "A: No, he was on for two years" in instruction
    assert "Current question: How long?" in instruction


def test_compose_instruction_no_passages() -> None:
    instruction = compose_instruction(
        prompt="Hello",
        dialogue_type="other",
        source_passages=[],
        qa_pairs=[],
    )
    assert "You are a helpful assistant" in instruction
    assert "Current question: Hello" in instruction


def test_retrieve_context_with_embeddings() -> None:
    """End-to-end: store passages, embed them, retrieve via RAG."""
    storage = InMemoryStorage()
    provider = FakeProvider()
    embedding_store = InMemoryEmbeddingStore()
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    runtime_config = RuntimeConfig(
        default_model="fake-model",
        embeddings_enabled=True,
        chat_embeddings_enabled=True,
        embedding_model="nomic-embed-text",
    )

    from llmasm.conversation.memory import store_source_passage, store_user_question, store_assistant_answer

    workspace_id = "workspace_test"
    from llmasm.graph.models import WorkspaceGraph
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))

    # Store and embed passages
    p1 = store_source_passage(
        workspace_id,
        "Dennis Farina was on Law & Order for two years.",
        storage=storage,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=embedding_store,
    )
    embedding_store.attach_item("memory_item", p1.id, p1)
    p2 = store_source_passage(
        workspace_id,
        "In 2004 he joined Law & Order playing Detective Joe Fontana.",
        storage=storage,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=embedding_store,
    )
    embedding_store.attach_item("memory_item", p2.id, p2)
    # Store a question + answer pair
    q1 = store_user_question(
        workspace_id,
        "Was he on for five years?",
        storage=storage,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=embedding_store,
    )
    embedding_store.attach_item("memory_item", q1.id, q1)
    a1 = store_assistant_answer(
        workspace_id,
        "No, he was on for two years.",
        storage=storage,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=embedding_store,
    )
    embedding_store.attach_item("memory_item", a1.id, a1)

    passages, qa_pairs, search_query = retrieve_context(
        workspace_id,
        "How long?",
        storage=storage,
        provider=provider,
        runtime_config=runtime_config,
        embedding_store=embedding_store,
    )

    # Should retrieve at least the relevant passage
    assert len(passages) > 0
    assert any("two years" in p for p in passages)

    # Q/A pairs may or may not be retrieved depending on embedding similarity
    # but the structure should be correct
    assert isinstance(qa_pairs, list)
    assert isinstance(search_query, str)


def test_retrieve_context_fallback_when_no_embeddings() -> None:
    """When chat_embeddings_enabled=False, retrieve_context should still work
    but returns empty results because NullEmbeddingStore returns no matches."""
    storage = InMemoryStorage()
    provider = FakeProvider()
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    runtime_config = RuntimeConfig(
        default_model="fake-model",
        embeddings_enabled=False,
        chat_embeddings_enabled=False,
    )

    from llmasm.graph.models import WorkspaceGraph
    workspace_id = "workspace_test2"
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))

    passages, qa_pairs, search_query = retrieve_context(
        workspace_id,
        "How long?",
        storage=storage,
        provider=provider,
        runtime_config=runtime_config,
        embedding_store=NullEmbeddingStore(),
    )

    assert passages == []
    assert qa_pairs == []
    assert search_query == "How long?"
