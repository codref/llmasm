"""Tests for the text chunking module."""

from __future__ import annotations

import pytest

from llmasm.chunking import SentenceTextChunker
from llmasm.providers.base import SimpleTokenizer


def _make_chunker(target: int = 10, overlap: int = 2) -> SentenceTextChunker:
    return SentenceTextChunker(SimpleTokenizer(), target_tokens=target, overlap_tokens=overlap)


def test_empty_text_returns_empty_chunks() -> None:
    chunker = _make_chunker()
    assert chunker.chunk("") == []
    assert chunker.chunk("   \n\n   ") == []


def test_single_sentence_single_chunk() -> None:
    chunker = _make_chunker(target=100)
    chunks = chunker.chunk("This is one sentence.", source_id="s1")
    assert len(chunks) == 1
    assert chunks[0].text == "This is one sentence."
    assert chunks[0].metadata["source_id"] == "s1"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["total_chunks"] == 1
    assert chunks[0].metadata["is_summary"] is False


def test_chunking_splits_long_text() -> None:
    chunker = _make_chunker(target=10)
    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    chunks = chunker.chunk(text, source_id="doc")
    assert len(chunks) >= 2
    # Chunk texts should not overlap with each other except for the configured overlap,
    # and concatenating them should recover the original text approximately.
    combined = " ".join(c.text for c in chunks)
    assert "First sentence here" in combined
    assert "Fourth sentence here" in combined
    for chunk in chunks:
        assert chunk.metadata["source_id"] == "doc"
        assert chunk.metadata["total_chunks"] == len(chunks)


def test_overlap_preserves_context() -> None:
    chunker = _make_chunker(target=10, overlap=5)
    text = "A long first sentence with many words. B long second sentence with many words."
    chunks = chunker.chunk(text, source_id="doc")
    assert len(chunks) == 2
    # The second chunk should start with words from the end of the first chunk
    first_words = set(chunks[0].text.lower().split())
    second_words = set(chunks[1].text.lower().split())
    assert first_words & second_words


def test_long_single_sentence_kept_as_chunk() -> None:
    chunker = _make_chunker(target=5)
    text = "ThisIsAVeryLongSentenceWithoutSpacesThatExceedsTheTarget."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap_tokens must be smaller"):
        SentenceTextChunker(SimpleTokenizer(), target_tokens=10, overlap_tokens=10)


def test_chunk_token_counts_are_non_negative() -> None:
    chunker = _make_chunker(target=10)
    chunks = chunker.chunk("One. Two. Three. Four.")
    for chunk in chunks:
        assert chunk.token_count > 0


def test_text_without_punctuation_falls_back_to_lines() -> None:
    chunker = _make_chunker(target=5)
    text = "line one\nline two\nline three"
    chunks = chunker.chunk(text)
    assert len(chunks) == 3
    assert chunks[0].text == "line one"
