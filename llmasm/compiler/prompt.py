"""Planner prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from llmasm.config import RuntimeConfig
from llmasm.graph.models import Goal, GoalAction
from llmasm.graph.registry import SchemaRegistry
from llmasm.providers.base import ModelInfo
from llmasm.storage.base import FewShotExample
from llmasm.tools.registry import ToolRegistry


@dataclass(frozen=True)
class PriorContext:
    """Text context item for planner prompts."""

    title: str
    text: str


def render_planner_prompt(
    *,
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
    models: list[ModelInfo],
    active_goal: Goal | None,
    goal_action: GoalAction,
    prior_context: Iterable[PriorContext],
    user_prompt: str,
    runtime_config: RuntimeConfig,
    few_shot_examples: list[FewShotExample] | None = None,
    repair_feedback: str | None = None,
) -> str:
    """Render a deterministic budget-aware planner prompt."""

    model_lines = "\n".join(
        f"- {model.name}: context_window={model.context_window}" for model in sorted(models, key=lambda m: m.name)
    )
    required = [
        "You are the LLMASM planner. Emit one TaskGraphProposal JSON object and no prose.",
        "Planning rules:\n"
        "- Always echo the exact active goal_action value.\n"
        "- For retrieval QA tasks, prefer a simple DAG: intent -> retrieval tool -> model -> final.\n"
        "- Include an intent node with output_schema RawText and metadata.output.text set to the user task.\n"
        "- Tool nodes must set execution.tool to one registered tool name.\n"
        "- Tool nodes must set input_schema/output_schema exactly as shown in the tool registry.\n"
        "- Tool, model, and final nodes must each have at least one incoming edge.\n"
        "- Model nodes should use output_schema Summary unless the tool output already is FinalAnswer.\n"
        "- QA model nodes should set metadata.instruction or metadata.description to the answer instruction.\n"
        "- Final nodes should use input_schema Summary and output_schema FinalAnswer.\n"
        "- Use edge ports named output and input unless a node declares explicit ports.",
        "Schemas:\n" + schema_registry.describe(),
        "Tools:\n" + (tool_registry.describe() or "- none"),
        "Models:\n" + (model_lines or "- none"),
        "Active goal:\n"
        + f"id={active_goal.id if active_goal else None}\n"
        + f"goal_action={goal_action.value}\n"
        + f"text={active_goal.text if active_goal else ''}",
        "User prompt:\n" + user_prompt,
    ]
    sections = list(required)
    budget = runtime_config.planner_max_tokens - runtime_config.repair_section_reserve
    for item in prior_context:
        candidate = f"Context: {item.title}\n{item.text}"
        if _fits(sections + [candidate], budget, runtime_config):
            sections.append(candidate)
    for example in few_shot_examples or []:
        candidate = f"Few-shot intent: {example.intent}\n{example.proposal_json}"
        if _fits(sections + [candidate], budget, runtime_config):
            sections.append(candidate)
    if repair_feedback:
        sections.append("Previous errors to fix:\n" + repair_feedback)
    return "\n\n---\n\n".join(sections)


def _fits(sections: list[str], budget: int, config: RuntimeConfig) -> bool:
    return config.tokenizer.count_tokens("\n".join(sections)) <= budget
