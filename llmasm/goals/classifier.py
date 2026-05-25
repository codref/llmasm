"""Deterministic goal classification."""

from __future__ import annotations

from llmasm.graph.models import Goal, GoalAction

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
    if word_overlap_ratio(prompt, active_goal.text) >= 0.25:
        return GoalAction.CONTINUE
    return GoalAction.NEW


def word_overlap_ratio(left: str, right: str) -> float:
    """Return overlap ratio from left into right."""

    left_words = {word for word in left.lower().split() if word}
    right_words = {word for word in right.lower().split() if word}
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words)
