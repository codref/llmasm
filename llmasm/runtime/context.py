"""Runtime context selection."""

from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel

from llmasm.config import RuntimeConfig
from llmasm.graph.models import Node, Run
from llmasm.storage.base import ContextItem, Storage
from llmasm.storage.embeddings import EmbeddingStore


@dataclass(frozen=True)
class SelectedContext:
    """Context selected for a node invocation."""

    items: list[ContextItem]
    direct_inputs: dict[str, BaseModel]


def select_context(
    *,
    storage: Storage,
    runtime_config: RuntimeConfig,
    run: Run,
    node: Node,
    direct_inputs: dict[str, BaseModel],
    embedding_store: EmbeddingStore,
) -> SelectedContext:
    """Select context with direct inputs first, then relevant workspace memory."""

    query = " ".join(
        [
            node.name,
            str(node.metadata.get("instruction", "")),
            " ".join(str(value.model_dump()) for value in direct_inputs.values()),
        ]
    )
    budget = int(node.metadata.get("max_input_tokens") or runtime_config.default_context_tokens)
    items: list[ContextItem] = []
    if runtime_config.embeddings_enabled:
        # The executor owns provider embedding calls; without a query vector available here,
        # vector search is deliberately skipped in the generic selector.
        items.extend([])
    if hasattr(storage, "retrieve_workspace_context"):
        items.extend(storage.retrieve_workspace_context(run.workspace_graph_id, query, budget, {}))  # type: ignore[attr-defined]
    ranked = sorted(items, key=lambda item: (item.score, -item.token_count), reverse=True)
    return SelectedContext(items=_trim(ranked, budget), direct_inputs=direct_inputs)


def _trim(items: list[ContextItem], budget: int) -> list[ContextItem]:
    total = 0
    kept: list[ContextItem] = []
    for item in items:
        if total + item.token_count > budget:
            continue
        total += item.token_count
        kept.append(item)
    return kept
