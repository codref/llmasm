"""Runtime context selection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from llmasm.config import RuntimeConfig
from llmasm.conversation.retrieval import prepare_search_query
from llmasm.graph.models import MemoryItem, Node, Run
from llmasm.providers.base import LLMProvider
from llmasm.storage.base import ContextItem, Storage
from llmasm.storage.embeddings import EmbeddingStore, ScoredMatch

_SUMMARY_RESERVE_TOKENS = 128

log = logging.getLogger(__name__)

_MAX_SNIPPET_CHARS = 200
_MAX_CANDIDATES_FOR_LLM = 12

# Memory kinds allowed in runtime context by default.
# user_question / assistant_answer are excluded to avoid the model re-answering
# prior turns. A node can opt-in via metadata["context_memory_kinds"].
DEFAULT_RUNTIME_CONTEXT_KINDS = {"source_passage", "human_note", "system_note"}


class ContextRelevanceFilter(BaseModel):
    """Structured output for LLM-based context relevance filtering."""

    relevant_ids: list[str] = Field(
        description="IDs of context items that are genuinely relevant to the query."
    )


def _relevance_filter_prompt(query: str, candidates: list[ContextItem]) -> str:
    lines = ["You are a relevance filter. Given a QUERY and a list of CONTEXT ITEMS, return only the IDs of items that are genuinely relevant to answering the query. Exclude items from unrelated topics even if they appear in the same workspace."]
    lines.append(f"\nQUERY:\n{query[:300]}")
    lines.append("\nCONTEXT ITEMS:")
    for item in candidates:
        snippet = item.text[:_MAX_SNIPPET_CHARS].replace("\n", " ")
        lines.append(f'  id={item.id}  text="{snippet}..."')
    lines.append('\nReturn ONLY valid JSON matching the schema: {"relevant_ids": ["<id>", ...]}')
    return "\n".join(lines)


def filter_context_with_llm(
    query: str,
    candidates: list[ContextItem],
    provider: LLMProvider,
    options: dict[str, Any] | None = None,
) -> list[ContextItem]:
    """Use an LLM call to discard irrelevant context items; fall back to returning all candidates on error."""

    if not candidates:
        return candidates
    # Cap candidates sent to LLM to avoid large prompts
    batch = candidates[:_MAX_CANDIDATES_FOR_LLM]
    try:
        output = provider.generate(
            _relevance_filter_prompt(query, batch),
            options or {},
            ContextRelevanceFilter.model_json_schema(),
        )
        raw = str(getattr(output, "text", output)).strip()
        result = ContextRelevanceFilter.model_validate_json(raw)
        kept_ids = set(result.relevant_ids)
        kept = [item for item in batch if item.id in kept_ids]
        # Append any items beyond the batch limit unchanged (they weren't reviewed)
        kept += candidates[_MAX_CANDIDATES_FOR_LLM:]
        return kept
    except (json.JSONDecodeError, ValidationError, Exception) as exc:
        log.debug("LLM context filter failed (%s); returning all candidates.", exc)
        return candidates


@dataclass(frozen=True)
class SelectedContext:
    """Context selected for a node invocation."""

    items: list[ContextItem]
    direct_inputs: dict[str, BaseModel]
    search_query: str = ""


def _allowed_memory_kinds(node: Node, runtime_config: RuntimeConfig) -> set[str]:
    """Return the memory kinds that may be retrieved for this node."""
    node_kinds = node.metadata.get("context_memory_kinds")
    if node_kinds:
        return set(node_kinds)
    config_kinds = runtime_config.runtime_context_memory_kinds
    if config_kinds is not None:
        return set(config_kinds)
    return DEFAULT_RUNTIME_CONTEXT_KINDS


def _direct_input_tokens(direct_inputs: dict[str, BaseModel], tokenizer) -> int:
    total = 0
    for value in direct_inputs.values():
        text = json.dumps(value.model_dump(mode="json"), sort_keys=True)
        total += max(1, tokenizer.count_tokens(text))
    return total


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
    allowed_kinds = _allowed_memory_kinds(node, runtime_config)

    # ── Short-circuit: if direct inputs are already rich enough, skip workspace retrieval ──
    direct_tokens = _direct_input_tokens(direct_inputs, runtime_config.tokenizer)
    sufficiency_threshold = runtime_config.context_sufficiency_threshold_tokens
    if runtime_config.prefer_run_context and sufficiency_threshold > 0 and direct_tokens >= sufficiency_threshold:
        return SelectedContext(items=[], direct_inputs=direct_inputs, search_query="")

    # Prefer a search query computed once for this run (e.g. by LLMASM.ask).
    # Fall back to computing it here for callers that invoke the executor directly.
    search_query = run.metadata.get("search_query") or prepare_search_query(
        query,
        run.workspace_graph_id,
        storage=storage,
        provider=provider,
        runtime_config=runtime_config,
    )

    # ── Vector-similarity items (when embeddings are enabled) ──────────────
    workspace_ids = [run.workspace_graph_id, *runtime_config.reference_workspace_ids]
    vector_items: list[ContextItem] = []
    if runtime_config.embeddings_enabled:
        output = provider.embed([search_query], {"model": runtime_config.embedding_model})[0]
        vector_matches = embedding_store.search_similar(
            output.vector,
            {"owner_type": "memory_item", "workspace_graph_ids": workspace_ids},
            limit=20,
        )
        vector_items = [
            _match_to_context_item(m, runtime_config.tokenizer)
            for m in vector_matches
            if isinstance(m.item, MemoryItem) and m.item.kind in allowed_kinds
        ]

    # ── Word-overlap items ─────────────────────────────────────────────────
    text_items: list[ContextItem] = []
    if hasattr(storage, "retrieve_workspace_context"):
        text_items = storage.retrieve_workspace_context(
            workspace_ids, search_query, budget, {}, kinds=allowed_kinds
        )

    # ── Merge: deduplicate by item id, keep highest score ─────────────────
    by_id: dict[str, ContextItem] = {}
    for ctx_item in vector_items + text_items:
        if ctx_item.id not in by_id or ctx_item.score > by_id[ctx_item.id].score:
            by_id[ctx_item.id] = ctx_item
    items = list(by_id.values())

    min_score = runtime_config.context_min_score

    # ── Split source passages into summaries and chunks ─────────────────────
    summary_items = [item for item in items if _is_source_summary(item)]
    chunk_items = sorted(
        [item for item in items if _is_source_chunk(item) and item.score >= min_score],
        key=lambda item: (item.score, -item.token_count),
        reverse=True,
    )
    other_items = sorted(
        [item for item in items if not _is_source_summary(item) and not _is_source_chunk(item) and item.score >= min_score],
        key=lambda item: (item.score, -item.token_count),
        reverse=True,
    )

    # Reserve a small budget for summaries and place them first.
    summary_budget = min(_SUMMARY_RESERVE_TOKENS, budget)
    ranked = _trim(summary_items, summary_budget) + chunk_items + other_items

    # ── LLM relevance filter (opt-in) ──────────────────────────────────────
    if runtime_config.llm_context_filter and ranked:
        ranked = filter_context_with_llm(
            search_query,
            ranked,
            provider,
            {"model": runtime_config.default_model},
        )

    return SelectedContext(items=_trim(ranked, budget), direct_inputs=direct_inputs, search_query=search_query)


def _is_source_summary(item: ContextItem) -> bool:
    underlying: object = item.item
    return (
        isinstance(underlying, MemoryItem)
        and underlying.kind == "source_passage"
        and underlying.metadata.get("is_summary") is True
    )


def _is_source_chunk(item: ContextItem) -> bool:
    underlying: object = item.item
    return (
        isinstance(underlying, MemoryItem)
        and underlying.kind == "source_passage"
        and underlying.metadata.get("is_summary") is not True
    )


def _trim(items: list[ContextItem], budget: int) -> list[ContextItem]:
    total = 0
    kept: list[ContextItem] = []
    for item in items:
        if total + item.token_count > budget:
            continue
        total += item.token_count
        kept.append(item)
    return kept
