"""Embedding storage and lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from typing import Any, Protocol

from llmasm.config import RuntimeConfig
from llmasm.graph.models import Artifact, EmbeddingRef, MemoryItem
from llmasm.ids import new_id
from llmasm.providers.base import LLMProvider


@dataclass(frozen=True)
class ScoredMatch:
    """A vector search match."""

    item: MemoryItem | Artifact
    score: float
    embedding_id: str


class EmbeddingStore(Protocol):
    """Embedding persistence and vector search contract."""

    def persist(self, ref: EmbeddingRef, vector: list[float]) -> None: ...
    def search_similar(
        self, query_vector: list[float], filters: dict[str, object] | None, limit: int
    ) -> list[ScoredMatch]: ...
    def find_by_owner(self, owner_type: str, owner_id: str) -> EmbeddingRef | None: ...
    def has_embedding(self, owner_type: str, owner_id: str, text_hash: str) -> bool: ...


class InMemoryEmbeddingStore:
    """Cosine-similarity embedding store for tests."""

    def __init__(self) -> None:
        self.refs: dict[str, EmbeddingRef] = {}
        self.vectors: dict[str, list[float]] = {}
        self.items: dict[str, MemoryItem | Artifact] = {}

    def attach_item(self, owner_type: str, owner_id: str, item: MemoryItem | Artifact) -> None:
        """Attach an item so searches can return rich matches."""

        self.items[f"{owner_type}:{owner_id}"] = item

    def persist(self, ref: EmbeddingRef, vector: list[float]) -> None:
        self.refs[ref.id] = ref
        self.vectors[ref.id] = vector

    def search_similar(
        self, query_vector: list[float], filters: dict[str, object] | None, limit: int
    ) -> list[ScoredMatch]:
        filters = filters or {}
        owner_type_filter = filters.get("owner_type")
        workspace_filter = filters.get("workspace_graph_id")
        matches: list[ScoredMatch] = []
        for embedding_id, ref in self.refs.items():
            if owner_type_filter and owner_type_filter != ref.owner_type:
                continue
            item = self.items.get(f"{ref.owner_type}:{ref.owner_id}")
            if item is None:
                continue
            if workspace_filter and isinstance(item, MemoryItem) and item.workspace_graph_id != workspace_filter:
                continue
            matches.append(
                ScoredMatch(item=item, score=_cosine(query_vector, self.vectors[embedding_id]), embedding_id=embedding_id)
            )
        return sorted(matches, key=lambda match: match.score, reverse=True)[:limit]

    def find_by_owner(self, owner_type: str, owner_id: str) -> EmbeddingRef | None:
        for ref in self.refs.values():
            if ref.owner_type == owner_type and ref.owner_id == owner_id:
                return ref
        return None

    def has_embedding(self, owner_type: str, owner_id: str, text_hash: str) -> bool:
        ref = self.find_by_owner(owner_type, owner_id)
        return ref is not None and ref.text_hash == text_hash


class NullEmbeddingStore:
    """No-op embedding store used when embeddings are disabled."""

    def persist(self, ref: EmbeddingRef, vector: list[float]) -> None:
        return None

    def search_similar(
        self, query_vector: list[float], filters: dict[str, object] | None, limit: int
    ) -> list[ScoredMatch]:
        return []

    def find_by_owner(self, owner_type: str, owner_id: str) -> EmbeddingRef | None:
        return None

    def has_embedding(self, owner_type: str, owner_id: str, text_hash: str) -> bool:
        return False


def embed_and_persist(
    text: str,
    owner_type: str,
    owner_id: str,
    runtime_config: RuntimeConfig,
    provider: LLMProvider,
    embedding_store: EmbeddingStore,
) -> EmbeddingRef | None:
    """Embed text when enabled and persist the reference plus vector."""

    if not runtime_config.embeddings_enabled:
        return None
    text_hash = sha256(text.encode("utf-8")).hexdigest()
    if embedding_store.has_embedding(owner_type, owner_id, text_hash):
        return embedding_store.find_by_owner(owner_type, owner_id)
    output = provider.embed([text], {"model": runtime_config.embedding_model})[0]
    ref = EmbeddingRef(
        id=new_id("memory"),
        owner_type=owner_type,  # type: ignore[arg-type]
        owner_id=owner_id,
        model=runtime_config.embedding_model,
        dimensions=len(output.vector),
        text_hash=text_hash,
    )
    embedding_store.persist(ref, output.vector)
    return ref


def write_memory_item(
    workspace_graph_id: str,
    kind: str,
    text: str,
    runtime_config: RuntimeConfig,
    provider: LLMProvider,
    embedding_store: EmbeddingStore,
    storage: Any,
    *,
    source_artifact_id: str | None = None,
    source_run_id: str | None = None,
    confidence: float = 1.0,
) -> MemoryItem:
    """Persist a MemoryItem and optionally embed its text.

    Args:
        storage: Any object implementing the Storage protocol
            (typed as Any to avoid circular imports).
    """
    item = MemoryItem(
        id=new_id("memory"),
        workspace_graph_id=workspace_graph_id,
        kind=kind,
        text=text,
        source_artifact_id=source_artifact_id,
        source_run_id=source_run_id,
        confidence=confidence,
    )
    storage.persist_memory_item(item)
    embed_and_persist(text, "memory_item", item.id, runtime_config, provider, embedding_store)
    return item


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    ln = sqrt(sum(a * a for a in left))
    rn = sqrt(sum(b * b for b in right))
    if ln == 0 or rn == 0:
        return 0.0
    return dot / (ln * rn)
