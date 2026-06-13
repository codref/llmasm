"""Planner proposal models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from llmasm.graph.models import GoalAction, NodeKind, WorkspaceEdgeType


class ProposalExecution(BaseModel):
    """Node execution configuration emitted by a planner."""

    provider: str | None = None
    model: str | None = None
    tool: str | None = None
    prompt_template: str | None = None
    allow_cache: bool | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ProposalPort(BaseModel):
    """Port declaration in a planner proposal."""

    name: str
    direction: Literal["input", "output"]
    schema_ref: str
    required: bool = True


class ProposalNode(BaseModel):
    """Node declaration in a planner proposal."""

    name: str
    kind: NodeKind
    input_schema: str | None = None
    output_schema: str | None = None
    ports: list[ProposalPort] = Field(default_factory=list)
    execution: ProposalExecution = Field(default_factory=ProposalExecution)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProposalEdge(BaseModel):
    """Task edge declaration in a planner proposal."""

    from_node: str
    from_port: str = "output"
    to_node: str
    to_port: str = "input"
    transform: str | None = None
    required: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceLinkProposal(BaseModel):
    """Workspace edge declaration in a planner proposal."""

    edge_type: WorkspaceEdgeType
    from_type: str
    from_id: str | None = None
    from_node: str | None = None
    to_type: str
    to_id: str | None = None
    to_node: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskGraphProposal(BaseModel):
    """Structured task graph proposal returned by the planner."""

    intent: str
    goal_action: GoalAction
    goal_update_text: str | None = None
    nodes: list[ProposalNode]
    edges: list[ProposalEdge] = Field(default_factory=list)
    workspace_links: list[WorkspaceLinkProposal] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
