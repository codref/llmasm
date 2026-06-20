"""Runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from llmasm.providers.base import SimpleTokenizer
from llmasm.tokenizers import Tokenizer


@dataclass
class RuntimeConfig:
    """Configuration shared by compiler and runtime."""

    planner_model: str = "llama3.1:8b"
    planner_max_tokens: int = 6144
    compiler_max_attempts: int = 5
    repair_section_reserve: int = 300
    tokenizer: Tokenizer = field(default_factory=SimpleTokenizer)
    scope: str = "local"
    default_model: str = "llama3.1:8b"
    default_context_tokens: int = 4096
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    embeddings_enabled: bool = False
    reference_workspace_ids: list[str] = field(default_factory=list)
    llm_goal_classifier: bool = False
    classifier_context_depth: int = 3
    classifier_goal_text_chars: int = 400
    context_min_score: float = 0.05
    llm_context_filter: bool = False
    conversation_fast_path: bool = True
    grounded_qa_strict: bool = True
    chat_embeddings_enabled: bool = False
    chat_qa_truncate_chars: int | None = None
    llm_query_rewrite: bool = False
    llm_dialogue_classifier: bool = False

    # Runtime context selection (planner / non-fast-path)
    runtime_context_memory_kinds: list[str] | None = None
    context_sufficiency_threshold_tokens: int = 0
    prefer_run_context: bool = True

    # Document chunking for long source passages
    chunking_enabled: bool = True
    chunking_trigger_tokens: int = 512
    chunk_target_tokens: int = 256
    chunk_overlap_tokens: int = 32
    chunking_summary_enabled: bool = True
