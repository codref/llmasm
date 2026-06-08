"""Tests for deterministic conversation classifier."""

from __future__ import annotations

import json

import pytest

from llmasm.conversation.classifier import (
    DialogueClassification,
    DialogueType,
    classify_dialogue,
    classify_dialogue_llm,
)
from tests.unit.fakes import FakeProvider


class TestClassifyDialogue:
    """Staten Island session classification expectations."""

    def test_setup_instruction(self) -> None:
        prompt = (
            "I will ask you several questions about the following passage. "
            "Please read it carefully and answer based on its content."
        )
        assert classify_dialogue(prompt, []) == DialogueType.INSTRUCTION

    def test_first_passage(self) -> None:
        prompt = (
            "Staten Island is one of the five boroughs of New York City in the U.S. state of New York..."
        )
        assert classify_dialogue(prompt, []) == DialogueType.SOURCE

    def test_second_passage(self) -> None:
        prompt = (
            "The North Shore—especially the neighborhoods of St. George, Tompkinsville, "
            "Clifton, and Stapleton—is the most urban part of the island; it contains the "
            "designated St. George Historic District and the St. Paul's Avenue-Stapleton "
            "Heights Historic District, which feature large Victorian houses. The East Shore "
            "is home to the F.D.R. Boardwalk, the fourth-longest in the world. The South "
            "Shore, site of the 17th-century Dutch and French Huguenot settlement, developed "
            "rapidly beginning in the 1960s and 1970s and is now mostly suburban in character. "
            "The West Shore is the least populated and most industrial part of the island."
        )
        assert classify_dialogue(prompt, []) == DialogueType.SOURCE

    def test_direct_question(self) -> None:
        prompt = "How many burroughs are there?"
        assert classify_dialogue(prompt, []) == DialogueType.QUESTION

    def test_followup_in_what_city(self) -> None:
        prompt = "in what city?"
        recent = ["How many burroughs are there?"]
        assert classify_dialogue(prompt, recent) == DialogueType.FOLLOWUP_QUESTION

    def test_followup_and_state(self) -> None:
        prompt = "and state?"
        recent = ["How many burroughs are there?", "in what city?"]
        assert classify_dialogue(prompt, recent) == DialogueType.FOLLOWUP_QUESTION

    def test_short_question_without_recent_is_question(self) -> None:
        prompt = "what city?"
        assert classify_dialogue(prompt, []) == DialogueType.QUESTION

    def test_very_short_prompt_no_question_mark_with_recent(self) -> None:
        prompt = "the capital"
        recent = ["what is the capital of France?"]
        assert classify_dialogue(prompt, recent) == DialogueType.FOLLOWUP_QUESTION

    def test_long_declarative_is_source(self) -> None:
        prompt = "The quick brown fox jumps over the lazy dog. " * 5
        assert classify_dialogue(prompt, []) == DialogueType.SOURCE

    def test_reading_comp_setup(self) -> None:
        prompt = "Answer the following questions based on the passage."
        assert classify_dialogue(prompt, []) == DialogueType.INSTRUCTION


class TestClassifyDialogueLLM:
    """LLM-backed dialogue classifier tests."""

    def test_llm_happy_path(self) -> None:
        """LLM returns valid JSON classification and token usage."""
        provider = FakeProvider(
            planner_outputs=[
                json.dumps({"type": "source", "reason": "Long declarative text."})
            ]
        )
        result, tokens = classify_dialogue_llm(
            "Staten Island is one of the five boroughs...",
            [],
            provider=provider,
            model="fake-model",
        )
        assert result == DialogueType.SOURCE
        assert tokens == 2  # FakeProvider returns 1 input + 1 output token

    def test_llm_fallback_on_bad_json(self) -> None:
        """LLM classifier falls back to heuristic when provider returns garbage."""
        provider = FakeProvider(planner_outputs=["not json"])
        result, tokens = classify_dialogue_llm(
            "Answer the following questions based on the passage.",
            [],
            provider=provider,
            model="fake-model",
        )
        # Heuristic would classify this as INSTRUCTION
        assert result == DialogueType.INSTRUCTION
        assert tokens == 0  # Fallback means 0 classification tokens

    def test_llm_fallback_on_exception(self) -> None:
        """LLM classifier falls back to heuristic when provider raises."""

        class BrokenProvider:
            def generate(self, prompt, options, format_schema):
                raise RuntimeError("boom")

        result, tokens = classify_dialogue_llm(
            "How many burroughs are there?",
            [],
            provider=BrokenProvider(),
            model="fake-model",
        )
        # Heuristic would classify this as QUESTION
        assert result == DialogueType.QUESTION
        assert tokens == 0  # Fallback means 0 classification tokens

    def test_llm_followup_classification(self) -> None:
        """LLM correctly classifies a follow-up question."""
        provider = FakeProvider(
            planner_outputs=[
                json.dumps({"type": "followup_question", "reason": "Short elliptical follow-up."})
            ]
        )
        result, tokens = classify_dialogue_llm(
            "and state?",
            ["How many burroughs are there?"],
            provider=provider,
            model="fake-model",
        )
        assert result == DialogueType.FOLLOWUP_QUESTION
        assert tokens == 2

    def test_llm_prompt_includes_recent_questions(self) -> None:
        """The classification prompt includes recent questions."""
        provider = FakeProvider(
            planner_outputs=[
                json.dumps({"type": "question", "reason": "Direct question."})
            ]
        )
        classify_dialogue_llm(
            "How many burroughs are there?",
            ["What is the capital?", "What is the population?"],
            provider=provider,
            model="fake-model",
        )
        prompt = provider.generate_prompts[-1]
        assert "Recent questions:" in prompt
        assert "What is the capital?" in prompt
        assert "What is the population?" in prompt
