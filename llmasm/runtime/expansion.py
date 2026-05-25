"""Runtime graph expansion support."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llmasm.compiler.proposal import ProposalEdge, ProposalNode
from llmasm.graph.models import Node, Run, RunNodeState, TaskEdge, WorkspaceEdge
from llmasm.graph.transforms import TransformRegistry
from llmasm.graph.validation import ValidationIssue, validate_acyclic, validate_edge_compatibility, validate_tools
from llmasm.ids import new_id
from llmasm.storage.base import Storage
from llmasm.tools.registry import ToolRegistry


class ExpansionRequest(BaseModel):
    """Request emitted by an expand node."""

    reason: str
    proposed_nodes: list[ProposalNode] = Field(default_factory=list)
    proposed_edges: list[ProposalEdge] = Field(default_factory=list)


class Expansion(BaseModel):
    """Applied expansion IDs."""

    created_node_ids: list[str]
    created_edge_ids: list[str]


def validate_expansion(
    *,
    run: Run,
    source_node: Node,
    request: ExpansionRequest,
    storage: Storage,
    tool_registry: ToolRegistry,
    transform_registry: TransformRegistry,
    max_nodes: int = 5,
) -> list[ValidationIssue]:
    """Validate an expansion before mutating the graph."""

    if not request.reason.strip():
        return [ValidationIssue("MISSING_WORKSPACE_LINK", source_node.name, "Expansion reason is required")]
    if len(request.proposed_nodes) > max_nodes:
        return [ValidationIssue("CONTEXT_BUDGET_EXCEEDED", source_node.name, "Too many expansion nodes")]
    graph = storage.load_task_graph(run.task_graph_id)
    # Shallow validation of proposed tool names and edge compatibility after applying.
    issues = []
    for proposed in request.proposed_nodes:
        if proposed.kind == "tool" and not tool_registry.has(str(proposed.execution.tool)):
            issues.append(ValidationIssue("UNKNOWN_TOOL", proposed.name, str(proposed.execution.tool)))
    if issues:
        return issues
    return validate_tools(graph, tool_registry) + validate_edge_compatibility(graph, transform_registry) + validate_acyclic(graph)


def apply_expansion(
    *,
    run: Run,
    source_node: Node,
    request: ExpansionRequest,
    storage: Storage,
) -> Expansion:
    """Persist new nodes, edges, and provenance for an accepted expansion."""

    graph = storage.load_task_graph(run.task_graph_id)
    created_nodes: list[str] = []
    name_to_id = {node.name: node.id for node in graph.nodes}
    for proposed in request.proposed_nodes:
        node_id = new_id("node")
        name_to_id[proposed.name] = node_id
        node = Node(
            id=node_id,
            workspace_graph_id=graph.workspace_graph_id,
            task_graph_id=graph.id,
            kind=proposed.kind,
            name=proposed.name,
            input_schema=proposed.input_schema,
            output_schema=proposed.output_schema,
            execution=proposed.execution.model_dump(exclude_none=True),
            metadata=proposed.metadata,
        )
        graph.nodes.append(node)
        storage.create_run_node_state(RunNodeState(run_id=run.id, node_id=node_id))
        created_nodes.append(node_id)
    created_edges: list[str] = []
    for proposed in request.proposed_edges:
        if proposed.from_node not in name_to_id or proposed.to_node not in name_to_id:
            continue
        edge = TaskEdge(
            id=new_id("edge"),
            workspace_graph_id=graph.workspace_graph_id,
            task_graph_id=graph.id,
            from_node_id=name_to_id[proposed.from_node],
            from_port=proposed.from_port,
            to_node_id=name_to_id[proposed.to_node],
            to_port=proposed.to_port,
            transform=proposed.transform,
            required=proposed.required,
            metadata=proposed.metadata,
        )
        graph.task_edges.append(edge)
        storage.persist_task_edge(edge)
        created_edges.append(edge.id)
    storage.persist_task_graph(graph)
    storage.persist_workspace_edge(
        WorkspaceEdge(
            id=new_id("edge"),
            workspace_graph_id=graph.workspace_graph_id,
            edge_type="expands_to",  # type: ignore[arg-type]
            from_type="node",
            from_id=source_node.id,
            to_type="task_graph",
            to_id=graph.id,
            reason=request.reason,
        )
    )
    return Expansion(created_node_ids=created_nodes, created_edge_ids=created_edges)
