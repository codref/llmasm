"""Structured conversation memory helpers."""

from __future__ import annotations


from llmasm.config import RuntimeConfig
from llmasm.graph.models import MemoryItem
from llmasm.providers.base import LLMProvider
from llmasm.storage.base import Storage
from llmasm.chunking import Chunk
from llmasm.storage.embeddings import EmbeddingStore, write_memory_item


def _attach_if_possible(item: MemoryItem, embedding_store: EmbeddingStore) -> None:
    """Attach item to InMemoryEmbeddingStore so search_similar can return it."""
    if hasattr(embedding_store, "attach_item"):
        embedding_store.attach_item("memory_item", item.id, item)


def store_source_passage(
    workspace_graph_id: str,
    text: str,
    *,
    storage: Storage,
    runtime_config: RuntimeConfig,
    provider: LLMProvider,
    embedding_store: EmbeddingStore,
    source_run_id: str | None = None,
    turn: int | None = None,
) -> MemoryItem:
    """Store an authoritative user-provided source passage."""
    item = write_memory_item(
        workspace_graph_id=workspace_graph_id,
        kind="source_passage",
        text=text,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=embedding_store,
        storage=storage,
        source_run_id=source_run_id,
        confidence=1.0,
        metadata={"turn": turn, "is_authoritative_source": True},
    )
    _attach_if_possible(item, embedding_store)
    return item


def store_source_passages(
    workspace_graph_id: str,
    chunks: list[Chunk],
    *,
    storage: Storage,
    runtime_config: RuntimeConfig,
    provider: LLMProvider,
    embedding_store: EmbeddingStore,
    source_run_id: str | None = None,
    turn: int | None = None,
) -> list[MemoryItem]:
    """Store multiple source-passage chunks produced by a chunker.

    Each chunk becomes its own ``source_passage`` memory item so that
    retrieval can surface only the relevant chunks instead of the whole
    document.
    """

    items: list[MemoryItem] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata.setdefault("turn", turn)
        metadata.setdefault("is_authoritative_source", True)
        item = write_memory_item(
            workspace_graph_id=workspace_graph_id,
            kind="source_passage",
            text=chunk.text,
            runtime_config=runtime_config,
            provider=provider,
            embedding_store=embedding_store,
            storage=storage,
            source_run_id=source_run_id,
            confidence=1.0,
            metadata=metadata,
        )
        _attach_if_possible(item, embedding_store)
        items.append(item)
    return items


def store_source_summary(
    workspace_graph_id: str,
    summary_text: str,
    *,
    source_id: str,
    storage: Storage,
    runtime_config: RuntimeConfig,
    provider: LLMProvider,
    embedding_store: EmbeddingStore,
    source_run_id: str | None = None,
    turn: int | None = None,
) -> MemoryItem:
    """Store a document-level summary linked to a chunked source."""

    item = write_memory_item(
        workspace_graph_id=workspace_graph_id,
        kind="source_passage",
        text=summary_text,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=embedding_store,
        storage=storage,
        source_run_id=source_run_id,
        confidence=1.0,
        metadata={
            "turn": turn,
            "source_id": source_id,
            "is_summary": True,
            "is_authoritative_source": True,
        },
    )
    _attach_if_possible(item, embedding_store)
    return item


def store_user_question(
    workspace_graph_id: str,
    text: str,
    *,
    storage: Storage,
    runtime_config: RuntimeConfig,
    provider: LLMProvider,
    embedding_store: EmbeddingStore,
    source_run_id: str | None = None,
    turn: int | None = None,
) -> MemoryItem:
    """Store a user question."""
    item = write_memory_item(
        workspace_graph_id=workspace_graph_id,
        kind="user_question",
        text=text,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=embedding_store,
        storage=storage,
        source_run_id=source_run_id,
        confidence=1.0,
        metadata={"turn": turn, "role": "user"},
    )
    _attach_if_possible(item, embedding_store)
    return item


def store_assistant_answer(
    workspace_graph_id: str,
    text: str,
    *,
    storage: Storage,
    runtime_config: RuntimeConfig,
    provider: LLMProvider,
    embedding_store: EmbeddingStore,
    source_run_id: str | None = None,
    turn: int | None = None,
) -> MemoryItem:
    """Store an assistant answer."""
    item = write_memory_item(
        workspace_graph_id=workspace_graph_id,
        kind="assistant_answer",
        text=text,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=embedding_store,
        storage=storage,
        source_run_id=source_run_id,
        confidence=1.0,
        metadata={"turn": turn, "role": "assistant"},
    )
    _attach_if_possible(item, embedding_store)
    return item


def store_system_note(
    workspace_graph_id: str,
    text: str,
    *,
    storage: Storage,
    runtime_config: RuntimeConfig,
    provider: LLMProvider,
    embedding_store: EmbeddingStore,
    source_run_id: str | None = None,
    turn: int | None = None,
) -> MemoryItem:
    """Store an optional injected system/human note."""
    item = write_memory_item(
        workspace_graph_id=workspace_graph_id,
        kind="system_note",
        text=text,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=embedding_store,
        storage=storage,
        source_run_id=source_run_id,
        confidence=1.0,
        metadata={"turn": turn, "role": "system"},
    )
    _attach_if_possible(item, embedding_store)
    return item


def list_conversation_memory(storage: Storage, workspace_graph_id: str) -> list[MemoryItem]:
    """Return all conversation memory items for a workspace, newest last."""
    items = storage.list_memory_items(workspace_graph_id)
    conversation_kinds = {"source_passage", "user_question", "assistant_answer", "system_note", "human_note"}
    items = [item for item in items if item.kind in conversation_kinds]
    items.sort(key=lambda x: x.created_at)
    return items


def get_source_passages(storage: Storage, workspace_graph_id: str) -> list[MemoryItem]:
    """Return authoritative source passages, newest first."""
    items = storage.list_memory_items(workspace_graph_id)
    sources = [item for item in items if item.kind == "source_passage"]
    sources.sort(key=lambda x: x.created_at, reverse=True)
    return sources


def get_recent_user_questions(storage: Storage, workspace_graph_id: str, limit: int = 5) -> list[str]:
    """Return recent user question texts, newest first."""
    items = storage.list_memory_items(workspace_graph_id)
    questions = [item for item in items if item.kind == "user_question"]
    questions.sort(key=lambda x: x.created_at, reverse=True)
    return [item.text for item in questions[:limit]]


def get_recent_qa_pairs(storage: Storage, workspace_graph_id: str, limit: int = 3) -> list[tuple[str, str]]:
    """Return recent (question, answer) pairs, newest first.

    Pairs are formed by temporal proximity: each question is matched with the
    assistant answer that was stored immediately after it.
    """
    items = storage.list_memory_items(workspace_graph_id)
    questions = sorted(
        [item for item in items if item.kind == "user_question"],
        key=lambda x: x.created_at,
        reverse=True,
    )
    answers = sorted(
        [item for item in items if item.kind == "assistant_answer"],
        key=lambda x: x.created_at,
        reverse=True,
    )

    pairs: list[tuple[str, str]] = []
    used_answers: set[str] = set()
    for q in questions:
        # Match with the most recent unused answer that came after this question
        matched = next(
            (a for a in answers if a.id not in used_answers and a.created_at >= q.created_at),
            None,
        )
        if matched:
            pairs.append((q.text, matched.text))
            used_answers.add(matched.id)
        if len(pairs) >= limit:
            break
    return pairs
