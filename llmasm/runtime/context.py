"""Runtime context selection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from llmasm.config import RuntimeConfig
from llmasm.graph.models import Artifact, MemoryItem, Node, Run
from llmasm.providers.base import LLMProvider
from llmasm.storage.base import ContextItem, Storage
from llmasm.storage.embeddings import EmbeddingStore, ScoredMatch

log = logging.getLogger(__name__)

_MAX_SNIPPET_CHARS = 200
_MAX_CANDIDATES_FOR_LLM = 12


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
        result = ContextRelevanceFilter.model_validate(json.loads(raw))
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
    min_score = runtime_config.context_min_score
    ranked = [item for item in ranked if item.score >= min_score]

    # ── LLM relevance filter (opt-in) ──────────────────────────────────────
    if runtime_config.llm_context_filter and ranked:
        ranked = filter_context_with_llm(
            query,
            ranked,
            provider,
            {"model": runtime_config.default_model},
        )

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
