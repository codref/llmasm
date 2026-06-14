"""Source-document ingestion helpers for planner mode."""

from __future__ import annotations

from dataclasses import dataclass

from llmasm.chunking import SentenceTextChunker
from llmasm.config import RuntimeConfig
from llmasm.conversation.classifier import DialogueType, classify_dialogue
from llmasm.conversation.memory import store_source_passages
from llmasm.ids import new_id
from llmasm.providers.base import LLMProvider
from llmasm.storage.base import Storage
from llmasm.storage.embeddings import EmbeddingStore, NullEmbeddingStore

SOURCE_PLACEHOLDER = "[A long source document has been stored in workspace chunks. Ask questions about it.]"


@dataclass(frozen=True)
class IngestionResult:
    """Result of ``maybe_ingest_long_source``."""

    effective_prompt: str
    source_id: str | None = None


def is_long_source_document(prompt: str, runtime_config: RuntimeConfig) -> bool:
    """Return True when ``prompt`` is a source document that should be chunked."""

    dialogue_type = classify_dialogue(prompt, recent_user_questions=[])
    if dialogue_type != DialogueType.SOURCE:
        return False
    tokens = runtime_config.tokenizer.count_tokens(prompt)
    return runtime_config.chunking_enabled and tokens > runtime_config.chunking_trigger_tokens


def maybe_ingest_long_source(
    workspace_graph_id: str,
    prompt: str,
    *,
    storage: Storage,
    provider: LLMProvider,
    runtime_config: RuntimeConfig,
    embedding_store: EmbeddingStore | None,
    turn: int | None = None,
) -> IngestionResult:
    """Chunk and store a long source prompt; return a placeholder for the planner.

    The prompt is only chunked when it is classified as a source document and
    its token count exceeds ``runtime_config.chunking_trigger_tokens``. Short
    prompts and non-source prompts are returned unchanged.

    Returns:
        An :class:`IngestionResult` containing the prompt that should be passed
        to the planner and, when chunking occurred, the ``source_id`` that links
        the chunks together so a later summary can be associated with them.
    """

    if not is_long_source_document(prompt, runtime_config):
        return IngestionResult(effective_prompt=prompt)

    source_id = new_id("memory")
    chunker = SentenceTextChunker(
        runtime_config.tokenizer,
        target_tokens=runtime_config.chunk_target_tokens,
        overlap_tokens=runtime_config.chunk_overlap_tokens,
    )
    chunks = chunker.chunk(prompt, source_id=source_id)
    store_source_passages(
        workspace_graph_id,
        chunks,
        storage=storage,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=embedding_store or NullEmbeddingStore(),
        turn=turn,
    )
    return IngestionResult(effective_prompt=SOURCE_PLACEHOLDER, source_id=source_id)
