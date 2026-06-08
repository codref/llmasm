"""Runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from llmasm.providers.base import SimpleTokenizer, TokenizerProtocol


@dataclass
class RuntimeConfig:
    """Configuration shared by compiler and runtime."""

    planner_model: str = "llama3.1:8b"
    planner_max_tokens: int = 6144
    compiler_max_attempts: int = 5
    repair_section_reserve: int = 300
    tokenizer: TokenizerProtocol = field(default_factory=SimpleTokenizer)
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
