"""Provider interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Protocol

from llmasm.tokenizers import Tokenizer


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a model available from a provider."""

    name: str
    context_window: int | None = None


@dataclass(frozen=True)
class ToolCallOutput:
    """A single tool call emitted by a model."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelOutput:
    """Text generation output."""

    text: str
    raw: dict[str, Any] | None = None
    token_usage: dict[str, int] | None = None
    tool_calls: list[ToolCallOutput] = field(default_factory=list)


@dataclass(frozen=True)
class EmbeddingOutput:
    """Embedding output for one or more texts."""

    vector: list[float]
    raw: dict[str, Any] | None = None


class TokenizerProtocol(Protocol):
    """Protocol for token estimation."""

    def count_tokens(self, text: str) -> int:
        """Return an estimated token count."""


class SimpleTokenizer(Tokenizer):
    """Default token estimator based on word count."""

    def count_tokens(self, text: str) -> int:
        """Return an inexpensive token estimate."""

        return ceil(len(text.split()) * 1.35)


class LLMProvider(Protocol):
    """Protocol implemented by model providers."""

    name: str

    def list_models(self) -> list[ModelInfo]:
        """Return available models."""

    def generate(
        self,
        prompt: str,
        options: dict[str, Any] | None = None,
        format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Generate text."""

    def embed(
        self,
        texts: list[str],
        options: dict[str, Any] | None = None,
    ) -> list[EmbeddingOutput]:
        """Embed texts."""
