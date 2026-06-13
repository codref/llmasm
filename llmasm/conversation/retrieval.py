"""Embedding-based context retrieval for the conversation fast path.

Supports pluggable query preparation (rewrite, HyDE, etc.) and
scoped retrieval within a single workspace.
"""

from __future__ import annotations

from typing import Protocol

from llmasm.config import RuntimeConfig
from llmasm.graph.models import MemoryItem
from llmasm.providers.base import LLMProvider
from llmasm.storage.base import Storage
from llmasm.storage.embeddings import EmbeddingStore, NullEmbeddingStore


class QueryPreparer(Protocol):
    """Transform a raw prompt into a search-ready query string."""

    def prepare(
        self,
        prompt: str,
        recent_qa_pairs: list[tuple[str, str]],
        provider: LLMProvider,
        model: str,
    ) -> str: ...


class PassthroughPreparer:
    """No-op preparer — returns the prompt unchanged."""

    def prepare(
        self,
        prompt: str,
        recent_qa_pairs: list[tuple[str, str]],
        provider: LLMProvider,
        model: str,
    ) -> str:
        return prompt


class LLMRewritePreparer:
    """Use a cheap LLM call to expand elliptical / follow-up prompts
    into standalone search queries.
    """

    def prepare(
        self,
        prompt: str,
        recent_qa_pairs: list[tuple[str, str]],
        provider: LLMProvider,
        model: str,
    ) -> str:
        if not recent_qa_pairs:
            return prompt
        # Build a compact context block from the most recent pair
        q, a = recent_qa_pairs[0]
        rewrite_prompt = (
            "The user just asked a follow-up question. "
            "Your job is to replace pronouns (he, she, it, which one, they, him, his) "
            "with the actual names or nouns from the prior answer.\n\n"
            "Rules:\n"
            "- Output ONLY the resolved question.\n"
            "- Do NOT explain, do NOT add notes, do NOT use brackets.\n"
            "- If the follow-up already has no pronouns, output it unchanged.\n\n"
            "Examples:\n"
            "Prior: He was on Law & Order.\n"
            "Follow-up: How long?\n"
            "Resolved: How long was Dennis Farina on Law & Order?\n\n"
            "Prior: The spinoff was Trial by Jury.\n"
            "Follow-up: Which one?\n"
            "Resolved: What was the name of the spinoff?\n\n"
            f"Prior: {_truncate_text(a, 200)}\n\n"
            f"Follow-up: {prompt}\n\n"
            f"Resolved:"
        )
        try:
            result = provider.generate(
                rewrite_prompt,
                {"model": model, "temperature": 0.0},
                None,
            )
            text = str(getattr(result, "text", "")).strip()
            # Basic sanity checks: must be non-empty, not contain instruction words
            if not text:
                return prompt
            if len(text) < len(prompt):
                return prompt
            # Reject meta-text
            lower = text.lower()
            if any(bad in lower for bad in {"requires", "please provide", "cannot answer", "no information", "context does not contain"}):
                return prompt
            return text
        except Exception:
            pass
        return prompt


class HyDEPreparer:
    """Hypothetical Document Embeddings placeholder.

    Generates a short hypothetical answer and uses *that* text as the
    search query. More powerful than simple rewrite, but costs more
    tokens. To enable, swap ``LLMRewritePreparer`` for ``HyDEPreparer``
    in the caller.
    """

    def prepare(
        self,
        prompt: str,
        recent_qa_pairs: list[tuple[str, str]],
        provider: LLMProvider,
        model: str,
    ) -> str:
        hyde_prompt = (
            "Write a short factual paragraph that answers the question. "
            "Use only the information you are certain about.\n\n"
            f"Question: {prompt}\n\n"
            "Answer:"
        )
        try:
            result = provider.generate(hyde_prompt, {"model": model, "temperature": 0.0}, None)
            text = str(getattr(result, "text", "")).strip()
            if text:
                return text
        except Exception:
            pass
        return prompt


def _truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Cut to last word boundary to avoid mid-word truncation
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated + "..."


def _default_preparer(runtime_config: RuntimeConfig) -> QueryPreparer:
    if runtime_config.llm_query_rewrite:
        return LLMRewritePreparer()
    return PassthroughPreparer()


def _is_summary_item(item: MemoryItem) -> bool:
    """Return True when a source_passage item is a document summary."""
    return bool(item.metadata.get("is_summary", False))


def retrieve_context(
    workspace_graph_id: str,
    prompt: str,
    *,
    storage: Storage,
    provider: LLMProvider,
    runtime_config: RuntimeConfig,
    embedding_store: EmbeddingStore | None = None,
    preparer: QueryPreparer | None = None,
) -> tuple[str | None, list[str], list[tuple[str, str]], str]:
    """Retrieve relevant passages and Q/A pairs for a prompt via embedding search.

    Returns a tuple of:
      - summary: document-level summary text, if one is stored
      - retrieved_passages: source chunk texts ranked by relevance
      - retrieved_qa_pairs: (question, answer) tuples ranked by relevance
      - search_query: the (possibly rewritten) query that was embedded
    """
    embedding_store = embedding_store or NullEmbeddingStore()
    prep = preparer or _default_preparer(runtime_config)

    # Gather recent Q/A pairs for rewrite
    from llmasm.conversation.memory import get_recent_qa_pairs

    recent_qa_pairs = get_recent_qa_pairs(storage, workspace_graph_id, limit=3)

    # 1. Prepare the search query (rewrite or HyDE)
    search_query = prep.prepare(
        prompt,
        recent_qa_pairs,
        provider,
        runtime_config.default_model,
    )

    # 2. Embed and search scoped to this workspace
    output = provider.embed([search_query], {"model": runtime_config.embedding_model})
    query_vector = output[0].vector

    matches = embedding_store.search_similar(
        query_vector,
        filters={
            "owner_type": "memory_item",
            "workspace_graph_id": workspace_graph_id,
        },
        limit=20,
    )

    # 3. Separate summaries, chunks, and conversation items
    summary_matches: list[tuple[MemoryItem, float]] = []
    chunk_matches: list[tuple[MemoryItem, float]] = []
    qa_matches: list[tuple[MemoryItem, float]] = []
    for match in matches:
        item = match.item
        if not isinstance(item, MemoryItem):
            continue
        if item.workspace_graph_id != workspace_graph_id:
            continue
        if item.kind == "source_passage":
            if _is_summary_item(item):
                summary_matches.append((item, match.score))
            else:
                chunk_matches.append((item, match.score))
        elif item.kind in {"user_question", "assistant_answer"}:
            qa_matches.append((item, match.score))

    # 4. Pick the highest-scoring summary (if any)
    summary: str | None = None
    if summary_matches:
        summary_matches.sort(key=lambda x: x[1], reverse=True)
        summary = summary_matches[0][0].text

    # 5. Build chunk passage list (deduplicated, sorted by score)
    seen_passages: set[str] = set()
    passages: list[str] = []
    chunk_matches.sort(key=lambda x: x[1], reverse=True)
    for item, _score in chunk_matches:
        if item.text not in seen_passages:
            passages.append(item.text)
            seen_passages.add(item.text)

    # 6. Build Q/A pair list (deduplicated, sorted by score)
    qa_pairs: list[tuple[str, str]] = []
    used_qa: set[str] = set()
    qa_matches.sort(key=lambda x: x[1], reverse=True)
    for item, _score in qa_matches:
        # Pair questions with subsequent answers by temporal proximity
        if item.kind == "user_question":
            # Find the closest answer after this question
            all_items = storage.list_memory_items(workspace_graph_id)
            candidates = [
                a for a in all_items
                if a.kind == "assistant_answer"
                and a.created_at >= item.created_at
                and a.text not in used_qa
            ]
            candidates.sort(key=lambda x: x.created_at)
            if candidates:
                answer_text = candidates[0].text
                max_chars = runtime_config.chat_qa_truncate_chars
                if max_chars is not None:
                    answer_text = _truncate_text(answer_text, max_chars)
                qa_pairs.append((item.text, answer_text))
                used_qa.add(candidates[0].text)
        if len(qa_pairs) >= 3:
            break

    return summary, passages, qa_pairs, search_query


def compose_instruction(
    prompt: str,
    dialogue_type: str,
    source_passages: list[str],
    qa_pairs: list[tuple[str, str]],
    summary: str | None = None,
) -> str:
    """Build the model prompt from an optional summary, retrieved passages and Q/A pairs."""
    parts: list[str] = []

    if source_passages:
        parts.append(
            "Answer using ONLY the source passages below. "
            "Base your answer on the information in the passages, including what they imply or contradict. "
            "If the passages contain no relevant information at all, say that the provided passage does not contain the answer. "
            "Do not use outside knowledge."
        )

        if summary:
            parts.append("\n--- Source summary ---")
            parts.append(summary)
            parts.append("---")

        parts.append("\n--- Source passages ---")
        for i, passage in enumerate(source_passages, 1):
            parts.append(f"Passage {i}:\n{passage}")
        parts.append("---\n")
    else:
        parts.append(
            "You are a helpful assistant. Answer the user's question based on the conversation context. "
            "If no relevant context is available, answer from your own knowledge."
        )

    if qa_pairs:
        parts.append("\n--- Relevant prior context ---")
        for q, a in qa_pairs[:3]:
            parts.append(f"Q: {q}\nA: {a}")
        parts.append("---\n")

    parts.append(f"\nCurrent question: {prompt}")
    return "\n".join(parts)
