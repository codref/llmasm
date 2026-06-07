"""Goal classification — deterministic heuristic and LLM-backed paths."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from llmasm.graph.models import Goal, GoalAction

if TYPE_CHECKING:
    from llmasm.providers.base import LLMProvider

log = logging.getLogger(__name__)

_MAX_GOAL_TEXT_CHARS = 200


class GoalClassification(BaseModel):
    """Structured output model for LLM-backed goal classification."""

    action: GoalAction
    reason: str


def _classification_prompt(prompt: str, active_goal: Goal) -> str:
    goal_text = active_goal.text[:_MAX_GOAL_TEXT_CHARS]
    return (
        "You are a goal-classification assistant. "
        "Classify the USER PROMPT relative to the ACTIVE GOAL and return JSON.\n\n"
        "Actions:\n"
        '  "new"      – the prompt starts an unrelated new goal.\n'
        '  "continue" – the prompt continues, deepens, or asks a follow-up about the current goal.\n'
        '  "steer"    – the prompt redirects or corrects the current goal.\n\n'
        "Classify regardless of the language of the prompts.\n\n"
        f"ACTIVE GOAL:\n{goal_text}\n\n"
        f"USER PROMPT:\n{prompt}\n\n"
        'Return ONLY valid JSON matching the schema: {"action": "<new|continue|steer>", "reason": "<one sentence>"}'
    )


def classify_goal_action_llm(
    prompt: str,
    active_goal: Goal | None,
    planner: LLMProvider,
    options: dict[str, Any] | None = None,
) -> GoalAction:
    """Classify using an LLM call; fall back to the deterministic classifier on any error."""

    if active_goal is None:
        return GoalAction.NEW
    try:
        output = planner.generate(
            _classification_prompt(prompt, active_goal),
            options or {},
            GoalClassification.model_json_schema(),
        )
        raw = str(getattr(output, "text", output)).strip()
        classification = GoalClassification.model_validate(json.loads(raw))
        return classification.action
    except (json.JSONDecodeError, ValidationError, Exception) as exc:
        log.debug("LLM goal classifier failed (%s); falling back to heuristic.", exc)
        return classify_goal_action(prompt, active_goal)

NEW_SIGNALS = [
    "new task",
    "different topic",
    "unrelated to",
    "start over",
    "forget everything",
    "new goal",
]
STEER_SIGNALS = [
    "instead",
    "actually",
    "wait",
    "forget that",
    "change the",
    "not that",
    "i meant",
    "correct that",
    "i was wrong",
    "let's focus on",
    "switch to",
]
CONTINUE_SIGNALS = [
    "now",
    "also",
    "next",
    "and then",
    "after that",
    "in addition",
    "furthermore",
    "what about",
    "can you also",
    "additionally",
    "on top of that",
    "in more detail",
    "more about",
    "tell me more",
    "explain more",
    "can you explain",
    "can you elaborate",
    "elaborate on",
    "going back",
    "related to",
    "in that context",
    "on that topic",
    "regarding",
    "what is the",
    "what are the",
    "how does",
    "how do",
    "why is",
    "why does",
]
REFERENCE_SIGNALS = [
    "it",
    "that",
    "this",
    "the previous",
    "the last",
    "the same",
    "the result",
    "the summary",
    "the conversation",
]


def classify_goal_action(prompt: str, active_goal: Goal | None) -> GoalAction:
    """Classify the prompt against the active goal using the RFC heuristic."""

    if active_goal is None:
        return GoalAction.NEW
    normalized = prompt.lower().strip()
    if any(signal in normalized for signal in NEW_SIGNALS):
        return GoalAction.NEW
    if any(signal in normalized for signal in STEER_SIGNALS):
        return GoalAction.STEER
    if any(signal in normalized for signal in CONTINUE_SIGNALS):
        return GoalAction.CONTINUE
    if any(signal in normalized for signal in REFERENCE_SIGNALS):
        return GoalAction.CONTINUE
    if word_overlap_ratio(prompt, active_goal.text) >= 0.12:
        return GoalAction.CONTINUE
    return GoalAction.NEW


def word_overlap_ratio(left: str, right: str) -> float:
    """Return overlap ratio from left into right."""

    left_words = {word for word in left.lower().split() if word}
    right_words = {word for word in right.lower().split() if word}
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words)
