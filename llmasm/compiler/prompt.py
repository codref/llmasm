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
        "- MANDATORY: Every task graph MUST contain exactly one node with kind=\"final\". No exceptions.\n"
        "- Allowed node kinds: intent, tool, model, compress, router, expand, final. Do NOT use goal, memory_query, or observation.\n"
        "- Always echo the exact active goal_action value.\n"
        "- The intent node is always the root: it has no incoming edges and no input_schema.\n"
        "- For retrieval QA tasks, prefer a simple DAG: intent -> retrieval tool -> model -> final.\n"
        "- If NO tools are listed in the Tools section, do not generate any tool nodes; answer directly with intent -> model -> final.\n"
        "- Include an intent node with output_schema RawText and metadata.output.text set to the user task.\n"
        "- Tool nodes must set execution.tool to one registered tool name exactly as listed in the Tools section.\n"
        "- Tool nodes must set input_schema/output_schema exactly as shown in the tool registry.\n"
        "- Tool, model, and final nodes must each have at least one incoming edge.\n"
        "- Model nodes should use output_schema Summary unless the tool output already is FinalAnswer.\n"
        "- QA model nodes should set metadata.instruction or metadata.description to the answer instruction.\n"
        "- When the user prompt is the exact placeholder '[A long source document has been stored in workspace chunks. Ask questions about it.]', include a model node named 'summarize_source' early in the graph. It should use input_schema RawText, output_schema Summary, and metadata.is_summary_node=true. Its instruction should ask for a concise summary of the stored source. Connect it to a downstream model node that answers the implicit question, then to the final node.\n"
        "- Final nodes MUST use input_schema Summary and output_schema FinalAnswer; connect the last model node's output to the final node's input.\n"
        "- Use edge ports named output and input unless a node declares explicit ports.\n"
        "- Port direction must be exactly 'input' or 'output' (not 'in'/'out').\n"
        "- Use a router node when:\n"
        "  (a) a tool might return NotFound and the two paths need different handling, OR\n"
        "  (b) the user explicitly asks to classify input and branch to different answer styles or outputs.\n"
        "  Router rules:\n"
        "  - Set output_schema to RoutingDecision.\n"
        "  - For deterministic routing (e.g. NotFound check) set execution.mode='deterministic', execution.default_branch and execution.not_found_branch.\n"
        "  - For model-assisted routing (e.g. user asks to classify and branch) set execution.mode='model', execution.model, and execution.branches (list of branch names).\n"
        "  - The router node's metadata.instruction must describe what criterion to use for routing.\n"
        "  - Every edge leaving a router node MUST include metadata: {\"branch\": \"<branch_name>\"} matching one of the declared branches.\n"
        "  - Each branch model node needs TWO incoming edges: one data edge from the intent node (carrying RawText) and one control edge from the router (carrying RoutingDecision, with branch metadata). The control edge from the router is used only for branch selection, not as data input.\n"
        "  - Each branch should have its own model node with a tailored instruction, converging to a shared final node.\n"
        "  - At least two distinct branch labels are required.\n"
        "  Example: user asks 'if history give narrative, if science give bullet points' → router with branches=[\"history\",\"science\"], two model nodes, one final.",
        "Schemas:\n" + schema_registry.describe(),
        "Tools:\n" + (tool_registry.describe() or "(no tools registered — do not generate tool nodes)"),
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
