"""Tokenizer base classes.

This module defines the extendable :class:`Tokenizer` base class. Concrete
implementations can subclass it and be plugged into components that need token
counts (e.g. chunkers, context budgeting).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract base class for token estimation.

    Subclasses only need to implement :meth:`count_tokens`. The class is kept
    dependency-free so model-specific tokenizers (tiktoken, Hugging Face,
    sentencepiece, etc.) can be added without pulling them in by default.
    """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return an estimated token count for ``text``."""
