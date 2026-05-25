"""Core graph and runtime models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(UTC)


class NodeKind(StrEnum):
    """Stable node kinds."""

    INTENT = "intent"
    TOOL = "tool"
    MODEL = "model"
    MEMORY_QUERY = "memory_query"
    COMPRESS = "compress"
    ROUTER = "router"
    EXPAND = "expand"
    GOAL = "goal"
    OBSERVATION = "observation"
    FINAL = "final"


class RunStatus(StrEnum):
    """Run status values."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(StrEnum):
    """Per-run node status values."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYABLE = "retryable"
    EXPANDED = "expanded"
    SKIPPED = "skipped"


class GoalAction(StrEnum):
    """Goal classification actions."""

    CONTINUE = "continue"
    STEER = "steer"
    NEW = "new"


class WorkspaceEdgeType(StrEnum):
    """Workspace semantic/provenance edge types."""

    DEPENDS_ON = "depends_on"
    PRODUCED = "produced"
    USED_CONTEXT = "used_context"
    SUMMARIZES = "summarizes"
    REFERS_TO = "refers_to"
    FOLLOWS_UP = "follows_up"
    SUPPORTS_GOAL = "supports_goal"
    CONTRADICTS = "contradicts"
    EXPANDS_TO = "expands_to"


class WorkspaceGraph(BaseModel):
    """Long-lived workspace graph metadata."""

    id: str
    name: str
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Port(BaseModel):
    """Typed input or output port on a node."""

    node_id: str | None = None
    name: str
    direction: Literal["input", "output"]
    schema_ref: str
    required: bool = True


class Node(BaseModel):
    """Executable or structural task graph node."""

    id: str
    workspace_graph_id: str
    task_graph_id: str
    kind: NodeKind
    name: str
    input_schema: str | None = None
    output_schema: str | None = None
    ports: list[Port] = Field(default_factory=list)
    execution: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    def output_port_name(self) -> str:
        """Return the preferred output port name for artifacts."""

        for port in self.ports:
            if port.direction == "output":
                return port.name
        return "output"


class TaskEdge(BaseModel):
    """Execution-scoped dataflow edge inside one task graph."""

    id: str
    workspace_graph_id: str
    task_graph_id: str
    from_node_id: str
    from_port: str
    to_node_id: str
    to_port: str
    transform: str | None = None
    required: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceEdge(BaseModel):
    """Semantic/provenance edge in a workspace graph."""

    id: str
    workspace_graph_id: str
    edge_type: WorkspaceEdgeType
    from_type: str
    from_id: str
    to_type: str
    to_id: str
    from_port: str | None = None
    to_port: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class TaskGraph(BaseModel):
    """Bounded executable subgraph."""

    id: str
    workspace_graph_id: str
    root_prompt_node_id: str | None = None
    parent_task_graph_id: str | None = None
    status: str = "compiled"
    compiler_version: str = "0.1.0"
    nodes: list[Node] = Field(default_factory=list)
    task_edges: list[TaskEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Run(BaseModel):
    """One execution of a task graph."""

    id: str
    workspace_graph_id: str
    task_graph_id: str
    status: RunStatus = RunStatus.PENDING
    program_counter_node_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RunNodeState(BaseModel):
    """Runtime state for one node in one run."""

    run_id: str
    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    attempts: int = 0
    last_error: dict[str, Any] | None = None
    output_artifact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utcnow)


class Artifact(BaseModel):
    """Persisted node output."""

    id: str
    run_id: str
    node_id: str
    port: str
    content_type: str = "application/json"
    content_json: Any = None
    content_ref: str | None = None
    token_count: int = 0
    superseded_by: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """Tool invocation trace."""

    id: str
    run_id: str
    node_id: str
    tool_name: str
    input_json: Any
    output_artifact_id: str | None = None
    status: str
    latency_ms: int | None = None
    created_at: datetime = Field(default_factory=utcnow)


class ModelCall(BaseModel):
    """Model invocation trace."""

    id: str
    run_id: str
    node_id: str
    provider: str
    model: str
    prompt_artifact_id: str | None = None
    output_artifact_id: str | None = None
    status: str
    token_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class MemoryItem(BaseModel):
    """Reusable persisted workspace memory."""

    id: str
    workspace_graph_id: str
    kind: str
    text: str
    source_artifact_id: str | None = None
    source_run_id: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class EmbeddingRef(BaseModel):
    """Identity and provenance for one embedding computation."""

    id: str
    owner_type: Literal["artifact", "memory_item", "node", "prompt"]
    owner_id: str
    model: str
    dimensions: int
    text_hash: str
    created_at: datetime = Field(default_factory=utcnow)


class Checkpoint(BaseModel):
    """Append-only execution checkpoint."""

    id: str
    run_id: str
    program_counter_node_id: str | None = None
    completed_node_ids: list[str] = Field(default_factory=list)
    failed_node_ids: list[str] = Field(default_factory=list)
    state_hash: str
    state_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Goal(BaseModel):
    """Workspace goal lifecycle record."""

    id: str
    workspace_graph_id: str
    text: str
    status: str = "provisional"
    active_task_graph_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
