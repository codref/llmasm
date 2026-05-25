"""Run analysis API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llmasm.graph.models import (
    Artifact,
    Checkpoint,
    ModelCall,
    NodeStatus,
    Run,
    RunNodeState,
    TaskEdge,
    TaskGraph,
    ToolCall,
    WorkspaceEdge,
    WorkspaceGraph,
)
from llmasm.storage.base import Storage


@dataclass(frozen=True)
class RunAnalysis:
    """Materialized view of one run."""

    workspace: WorkspaceGraph
    task_graph: TaskGraph
    run: Run
    task_edges: list[TaskEdge]
    workspace_edges: list[WorkspaceEdge]
    node_states: list[RunNodeState]
    artifacts: list[Artifact]
    tool_calls: list[ToolCall]
    model_calls: list[ModelCall]
    checkpoints: list[Checkpoint]

    def failed_nodes(self) -> list[RunNodeState]:
        """Return failed node states."""

        return [state for state in self.node_states if state.status == NodeStatus.FAILED]

    def token_usage(self) -> dict[str, int]:
        """Return aggregate token usage."""

        usage = {"input_tokens": 0, "output_tokens": 0, "artifact_tokens": 0}
        for artifact in self.artifacts:
            usage["artifact_tokens"] += artifact.token_count
        for call in self.model_calls:
            usage["input_tokens"] += int(call.token_json.get("input_tokens", 0))
            usage["output_tokens"] += int(call.token_json.get("output_tokens", 0))
        return usage

    def context_used_by_model_call(self, model_call_id: str) -> dict[str, Any]:
        """Return prompt artifact content for a model call."""

        call = next(item for item in self.model_calls if item.id == model_call_id)
        if call.prompt_artifact_id is None:
            return {}
        artifact = next(item for item in self.artifacts if item.id == call.prompt_artifact_id)
        return artifact.content_json if isinstance(artifact.content_json, dict) else {"value": artifact.content_json}

    def expansions_for_run(self) -> list[WorkspaceEdge]:
        """Return expansion provenance edges."""

        return [edge for edge in self.workspace_edges if edge.edge_type == "expands_to"]

    def follow_up_chain(self) -> list[WorkspaceEdge]:
        """Return follow-up workspace edges."""

        return [edge for edge in self.workspace_edges if edge.edge_type == "follows_up"]


def query_run(storage: Storage, run_id: str) -> RunAnalysis:
    """Load all persisted analysis data for a run."""

    run = storage.load_run(run_id)
    task_graph = storage.load_task_graph(run.task_graph_id)
    workspace = storage.load_workspace_graph(run.workspace_graph_id)
    return RunAnalysis(
        workspace=workspace,
        task_graph=task_graph,
        run=run,
        task_edges=storage.list_task_edges(task_graph.id),
        workspace_edges=storage.load_workspace_edges_for_task(task_graph.id),
        node_states=storage.list_run_node_states(run_id),
        artifacts=storage.list_artifacts(run_id),
        tool_calls=storage.list_tool_calls(run_id),
        model_calls=storage.list_model_calls(run_id),
        checkpoints=storage.list_checkpoints(run_id),
    )
