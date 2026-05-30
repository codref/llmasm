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
