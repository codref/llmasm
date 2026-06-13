"""Text chunking utilities for long-document RAG.

The public API is intentionally small:

* :class:`Chunk` – a chunk of text with token count and metadata.
* :class:`TextChunker` – abstract base class for chunkers.
* :class:`SentenceTextChunker` – sentence-aware chunker with optional overlap.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from llmasm.tokenizers import Tokenizer


@dataclass
class Chunk:
    """A text chunk produced by a chunker."""

    text: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TextChunker(ABC):
    """Abstract base class for text chunkers.

    Subclasses implement :meth:`chunk` and accept a :class:`~llmasm.tokenizers.Tokenizer`
    instance so token counts stay consistent across the application.
    """

    @abstractmethod
    def chunk(self, text: str, *, source_id: str | None = None) -> list[Chunk]:
        """Split ``text`` into chunks.

        Args:
            text: The text to split.
            source_id: Optional identifier for the original document; chunkers
                should include it in chunk metadata when provided.

        Returns:
            A list of chunks ordered by original position.
        """


class SentenceTextChunker(TextChunker):
    """Chunk text by sentences, merging short sentences until a token target.

    Chunks are allowed to exceed ``target_tokens`` when a single sentence is
    longer than the target, but this is rare for typical sentence-split text.
    Overlap is implemented by copying trailing sentences from the previous chunk
    into the start of the next chunk.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        target_tokens: int = 256,
        overlap_tokens: int = 32,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")

        self.tokenizer = tokenizer
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str, *, source_id: str | None = None) -> list[Chunk]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        index = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.count_tokens(sentence)

            # If adding this sentence would exceed the target and we already
            # have sentences, emit the current chunk and start a new one.
            if current_tokens and current_tokens + sentence_tokens > self.target_tokens:
                chunks.append(
                    self._make_chunk(current_sentences, source_id=source_id, index=index)
                )
                index += 1

                # Build overlap buffer from the tail of the emitted chunk.
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for previous in reversed(current_sentences):
                    previous_tokens = self.tokenizer.count_tokens(previous)
                    if overlap_tokens + previous_tokens > self.overlap_tokens:
                        break
                    overlap_sentences.insert(0, previous)
                    overlap_tokens += previous_tokens

                current_sentences = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        if current_sentences:
            chunks.append(
                self._make_chunk(current_sentences, source_id=source_id, index=index)
            )

        # Fill in total chunk count so callers can reconstruct ordering.
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_chunks"] = total

        return chunks

    def _make_chunk(
        self, sentences: list[str], *, source_id: str | None, index: int
    ) -> Chunk:
        text = " ".join(sentences)
        return Chunk(
            text=text,
            token_count=self.tokenizer.count_tokens(text),
            metadata={
                "source_id": source_id,
                "chunk_index": index,
                "is_summary": False,
            },
        )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving punctuation.

    Falls back to splitting on newlines for texts without sentence-ending
    punctuation so the chunker always makes progress.
    """
    stripped = text.strip()
    if not stripped:
        return []

    # Split after . ! ? followed by whitespace.
    raw = re.split(r"(?<=[.!?])\s+", stripped)
    sentences = [sentence.strip() for sentence in raw if sentence.strip()]

    # If there were no sentence delimiters, fall back to line breaks.
    if not sentences:
        sentences = [line.strip() for line in stripped.splitlines() if line.strip()]
        return sentences

    # A single "sentence" that still contains hard line breaks is effectively
    # unpunctuated text; split it into lines so the chunker can make progress.
    if len(sentences) == 1 and "\n" in sentences[0]:
        return [line.strip() for line in sentences[0].splitlines() if line.strip()]

    return sentences
