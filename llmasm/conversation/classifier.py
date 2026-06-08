"""Deterministic dialogue classifier for conversation mode."""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)


class DialogueType(StrEnum):
    """Classification of a user prompt in a conversation."""

    SOURCE = "source"
    QUESTION = "question"
    FOLLOWUP_QUESTION = "followup_question"
    INSTRUCTION = "instruction"
    OTHER = "other"


class DialogueClassification(BaseModel):
    """Structured output model for LLM-backed dialogue classification."""

    type: DialogueType
    reason: str


# Heuristic signals
_READING_COMPREHENSION_SETUP = [
    "i will ask you",
    "please read",
    "based on the following",
    "answer the following",
    "read the passage",
    "read the following",
    "answer based on",
    "answer the questions",
    "based on the passage",
    "based on the text",
]

_FOLLOWUP_ANAPHORA = [
    "and ",
    "what ",
    "in what ",
    "how ",
    "why ",
    "when ",
    "where ",
    "who ",
    "which ",
]

_SHORT_FOLLOWUP_MAX_CHARS = 40


def classify_dialogue(prompt: str, recent_user_questions: list[str]) -> DialogueType:
    """Classify a user prompt deterministically.

    Args:
        prompt: The raw user prompt.
        recent_user_questions: Recent questions from the user (newest first).
            Used to resolve elliptical follow-ups.
    """
    normalized = prompt.lower().strip()
    stripped = normalized.rstrip("?")

    # Reading-comprehension setup language takes priority over source
    if any(signal in normalized for signal in _READING_COMPREHENSION_SETUP):
        return DialogueType.INSTRUCTION

    # Long declarative text is likely source when there is prior instruction context
    word_count = len(stripped.split())
    if word_count >= 12 and not normalized.endswith("?"):
        return DialogueType.SOURCE

    # Direct question
    if normalized.endswith("?"):
        # Very short + anaphora/conjunctions + recent question context = follow-up
        if len(prompt) <= _SHORT_FOLLOWUP_MAX_CHARS and recent_user_questions:
            if any(stripped.startswith(signal) for signal in _FOLLOWUP_ANAPHORA):
                return DialogueType.FOLLOWUP_QUESTION
            # Catch things like "in what city?" or "and state?"
            if " " in stripped and len(stripped.split()) <= 4:
                return DialogueType.FOLLOWUP_QUESTION
        return DialogueType.QUESTION

    # Short declarative after recent question context may be an implicit follow-up
    if len(prompt) <= _SHORT_FOLLOWUP_MAX_CHARS and recent_user_questions and word_count <= 4:
        return DialogueType.FOLLOWUP_QUESTION

    # Task setup / steering without question mark
    if any(signal in normalized for signal in _READING_COMPREHENSION_SETUP):
        return DialogueType.INSTRUCTION

    return DialogueType.OTHER


def _classification_prompt(prompt: str, recent_questions: list[str]) -> str:
    recent_section = ""
    if recent_questions:
        recent_section = "\nRecent questions:\n" + "\n".join(f"- {q}" for q in recent_questions[:3]) + "\n"
    return (
        "You are a dialogue-classification assistant. "
        "Classify the USER INPUT into one of the categories below and return JSON.\n\n"
        "Categories:\n"
        "  'source'          – long declarative text that is a passage/document to be stored for later questions.\n"
        "  'question'        – a direct question asking for information.\n"
        "  'followup_question' – a short elliptical follow-up (e.g., 'and state?', 'how long?') that depends on a recent question.\n"
        "  'instruction'     – a meta directive (e.g., 'I will ask you questions about the following passage').\n"
        "  'other'           – anything else.\n\n"
        "Classify regardless of the language of the user input.\n"
        f"{recent_section}"
        f"USER INPUT:\n{prompt}\n\n"
        'Return ONLY valid JSON matching the schema: {"type": "<source|question|followup_question|instruction|other>", "reason": "<one sentence>"}'
    )


def classify_dialogue_llm(
    prompt: str,
    recent_user_questions: list[str],
    provider: Any,
    model: str,
    temperature: float = 0.0,
) -> tuple[DialogueType, int]:
    """Classify a user prompt using an LLM call.

    Falls back to the deterministic heuristic classifier on any error.

    Returns:
        A tuple of (dialogue_type, classification_tokens).
        classification_tokens is the total token usage from the classifier LLM call
        (input + output), or 0 when falling back to the heuristic.

    Args:
        prompt: The raw user prompt.
        recent_user_questions: Recent questions from the user (newest first).
        provider: An LLMProvider-compatible object with a ``generate`` method.
        model: Model name to use for the classification call.
        temperature: Sampling temperature for the classifier LLM.
    """
    try:
        options: dict[str, Any] = {"model": model, "temperature": temperature}
        output = provider.generate(
            _classification_prompt(prompt, recent_user_questions),
            options,
            DialogueClassification.model_json_schema(),
        )
        raw = str(getattr(output, "text", output)).strip()
        classification = DialogueClassification.model_validate_json(raw)
        usage = getattr(output, "token_usage", None) or {}
        total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return classification.type, total_tokens
    except (json.JSONDecodeError, ValidationError, Exception) as exc:
        log.debug("LLM dialogue classifier failed (%s); falling back to heuristic.", exc)
        return classify_dialogue(prompt, recent_user_questions), 0
