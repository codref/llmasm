"""Runtime context selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pydantic import BaseModel

from llmasm.config import RuntimeConfig
from llmasm.graph.models import Artifact, MemoryItem, Node, Run
from llmasm.providers.base import LLMProvider
from llmasm.storage.base import ContextItem, Storage
from llmasm.storage.embeddings import EmbeddingStore, ScoredMatch


@dataclass(frozen=True)
class SelectedContext:
    """Context selected for a node invocation."""

    items: list[ContextItem]
    direct_inputs: dict[str, BaseModel]


def _match_to_context_item(match: ScoredMatch, tokenizer) -> ContextItem:
    item = match.item
    if isinstance(item, MemoryItem):
        text = item.text
        kind = "memory_item"
        item_id = item.id
    else:  # Artifact
        text = json.dumps(item.content_json) if item.content_json else ""
        kind = "artifact"
        item_id = item.id
    tokens = max(1, tokenizer.count_tokens(text))
    return ContextItem(id=item_id, kind=kind, text=text, score=match.score, token_count=tokens, item=item)


def select_context(
    *,
    storage: Storage,
    runtime_config: RuntimeConfig,
    run: Run,
    node: Node,
    direct_inputs: dict[str, BaseModel],
    embedding_store: EmbeddingStore,
    provider: LLMProvider,
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

    # ── Vector-similarity items (when embeddings are enabled) ──────────────
    workspace_ids = [run.workspace_graph_id, *runtime_config.reference_workspace_ids]
    vector_items: list[ContextItem] = []
    if runtime_config.embeddings_enabled:
        output = provider.embed([query], {"model": runtime_config.embedding_model})[0]
        vector_matches = embedding_store.search_similar(
            output.vector,
            {"owner_type": "memory_item", "workspace_graph_ids": workspace_ids},
            limit=20,
        )
        vector_items = [_match_to_context_item(m, runtime_config.tokenizer) for m in vector_matches]

    # ── Word-overlap items ─────────────────────────────────────────────────
    text_items: list[ContextItem] = []
    if hasattr(storage, "retrieve_workspace_context"):
        text_items = storage.retrieve_workspace_context(workspace_ids, query, budget, {})  # type: ignore[attr-defined]

    # ── Merge: deduplicate by item id, keep highest score ─────────────────
    by_id: dict[str, ContextItem] = {}
    for ctx_item in vector_items + text_items:
        if ctx_item.id not in by_id or ctx_item.score > by_id[ctx_item.id].score:
            by_id[ctx_item.id] = ctx_item
    items = list(by_id.values())

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
