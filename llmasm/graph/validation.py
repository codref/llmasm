"""Pure task graph validation."""

from __future__ import annotations

from dataclasses import dataclass

from llmasm.compiler.proposal import TaskGraphProposal
from llmasm.graph.models import NodeKind, TaskEdge, TaskGraph, WorkspaceEdgeType
from llmasm.graph.registry import SchemaRegistry
from llmasm.graph.transforms import TransformRegistry
from llmasm.providers.base import ModelInfo
from llmasm.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ValidationIssue:
    """Structured validation issue."""

    code: str
    node_name: str | None
    detail: str


def validate_required_ports(task_graph: TaskGraph) -> list[ValidationIssue]:
    """Validate required input ports have an incoming edge."""

    issues: list[ValidationIssue] = []
    incoming = {(edge.to_node_id, edge.to_port) for edge in task_graph.task_edges}
    nodes = {node.id: node for node in task_graph.nodes}
    input_required_kinds = {NodeKind.TOOL, NodeKind.MODEL, NodeKind.COMPRESS, NodeKind.FINAL}
    for node in task_graph.nodes:
        if node.kind == NodeKind.INTENT:
            continue
        node_incoming = [edge for edge in task_graph.task_edges if edge.to_node_id == node.id]
        if node.kind in input_required_kinds and not node_incoming:
            issues.append(
                ValidationIssue(
                    "PORT_UNSATISFIED",
                    node.name,
                    "Node requires at least one incoming input edge",
                )
            )
        for port in node.ports:
            if port.direction == "input" and port.required and (node.id, port.name) not in incoming:
                issues.append(
                    ValidationIssue("PORT_UNSATISFIED", node.name, f"Missing input port {port.name}")
                )
    for edge in task_graph.task_edges:
        if edge.from_node_id not in nodes or edge.to_node_id not in nodes:
            issues.append(ValidationIssue("PORT_UNSATISFIED", None, f"Edge {edge.id} references unknown node"))
            continue
        src = nodes[edge.from_node_id]
        dst = nodes[edge.to_node_id]
        if not _has_port(src, edge.from_port, "output"):
            issues.append(
                ValidationIssue("PORT_UNSATISFIED", src.name, f"Missing output port {edge.from_port}")
            )
        if not _has_port(dst, edge.to_port, "input"):
            issues.append(
                ValidationIssue("PORT_UNSATISFIED", dst.name, f"Missing input port {edge.to_port}")
            )
    return issues


def validate_schema_refs(task_graph: TaskGraph, schemas: SchemaRegistry) -> list[ValidationIssue]:
    """Validate all schema tags are registered."""

    issues: list[ValidationIssue] = []
    for node in task_graph.nodes:
        for tag in (node.input_schema, node.output_schema):
            if tag and not schemas.has(tag):
                issues.append(ValidationIssue("UNKNOWN_SCHEMA", node.name, f"Unknown schema {tag}"))
        for port in node.ports:
            if not schemas.has(port.schema_ref):
                issues.append(ValidationIssue("UNKNOWN_SCHEMA", node.name, f"Unknown schema {port.schema_ref}"))
    return issues


def validate_tool_outputs_consumed(task_graph: TaskGraph) -> list[ValidationIssue]:
    """Validate tool node outputs are wired to a downstream consumer.

    A tool node whose output is never used wastes the tool call and often
    indicates a planner wiring mistake (e.g. intent -> tool and intent -> model
    -> final, with the tool left as a disconnected sink).
    """

    issues: list[ValidationIssue] = []
    nodes_by_id = {node.id: node for node in task_graph.nodes}
    outgoing_by_node: dict[str, list[TaskEdge]] = {node.id: [] for node in task_graph.nodes}
    for edge in task_graph.task_edges:
        outgoing_by_node.setdefault(edge.from_node_id, []).append(edge)

    consumable_kinds = {NodeKind.MODEL, NodeKind.COMPRESS, NodeKind.FINAL}
    for node in task_graph.nodes:
        if node.kind != NodeKind.TOOL:
            continue
        outgoing = outgoing_by_node.get(node.id, [])
        if not outgoing:
            issues.append(
                ValidationIssue(
                    "TOOL_OUTPUT_UNCONSUMED",
                    node.name,
                    f"Tool node '{node.name}' has no outgoing edges; its output is never used",
                )
            )
            continue
        consumed = False
        for edge in outgoing:
            target = nodes_by_id.get(edge.to_node_id)
            if target is not None and target.kind in consumable_kinds:
                consumed = True
                break
        if not consumed:
            issues.append(
                ValidationIssue(
                    "TOOL_OUTPUT_UNCONSUMED",
                    node.name,
                    f"Tool node '{node.name}' output is not connected to a model, compress, or final node",
                )
            )
    return issues


def validate_edge_compatibility(
    task_graph: TaskGraph, transforms: TransformRegistry
) -> list[ValidationIssue]:
    """Validate edge schemas are directly compatible or transformable."""

    issues: list[ValidationIssue] = []
    nodes = {node.id: node for node in task_graph.nodes}
    for edge in task_graph.task_edges:
        src = nodes.get(edge.from_node_id)
        dst = nodes.get(edge.to_node_id)
        if src is None or dst is None:
            continue
        # Edges leaving a router are control/branch edges — the RoutingDecision
        # artifact is not used as data input by the downstream node.
        if src.kind == NodeKind.ROUTER:
            continue
        from_schema = _port_schema(src, edge.from_port, "output") or src.output_schema
        to_schema = _port_schema(dst, edge.to_port, "input") or dst.input_schema
        if not from_schema or not to_schema:
            continue
        if edge.transform and edge.transform not in transforms._transforms:
            issues.append(ValidationIssue("UNKNOWN_TRANSFORM", dst.name, edge.transform))
        elif not transforms.can_transform(from_schema, to_schema, edge.transform):
            issues.append(
                ValidationIssue(
                    "SCHEMA_MISMATCH",
                    dst.name,
                    f"{from_schema} cannot connect to {to_schema}",
                )
            )
    return issues


def validate_tools(task_graph: TaskGraph, tools: ToolRegistry) -> list[ValidationIssue]:
    """Validate tool nodes reference registered tools."""

    issues = []
    for node in task_graph.nodes:
        if node.kind != NodeKind.TOOL:
            continue
        tool_name = str(node.execution.get("tool"))
        if not tools.has(tool_name):
            issues.append(ValidationIssue("UNKNOWN_TOOL", node.name, tool_name))
            continue
        spec = tools.get(tool_name).spec()
        if node.input_schema != spec.input_schema:
            issues.append(
                ValidationIssue(
                    "TOOL_SCHEMA_MISMATCH",
                    node.name,
                    f"Expected input_schema {spec.input_schema}, got {node.input_schema}",
                )
            )
        if node.output_schema != spec.output_schema:
            issues.append(
                ValidationIssue(
                    "TOOL_SCHEMA_MISMATCH",
                    node.name,
                    f"Expected output_schema {spec.output_schema}, got {node.output_schema}",
                )
            )
    return issues


def validate_models(task_graph: TaskGraph, models: list[ModelInfo]) -> list[ValidationIssue]:
    """Validate model nodes reference known models when a model list is provided."""

    known = {model.name for model in models}
    if not known:
        return []
    issues = []
    for node in task_graph.nodes:
        if node.kind in {NodeKind.MODEL, NodeKind.COMPRESS}:
            model = str(node.execution.get("model") or "")
            if model and model not in known:
                issues.append(ValidationIssue("UNKNOWN_MODEL", node.name, model))
    return issues


def validate_context_budgets(task_graph: TaskGraph) -> list[ValidationIssue]:
    """Validate node context budgets are positive."""

    issues = []
    for node in task_graph.nodes:
        budget = node.metadata.get("max_input_tokens") or node.execution.get("max_input_tokens")
        if budget is not None and int(budget) <= 0:
            issues.append(ValidationIssue("CONTEXT_BUDGET_EXCEEDED", node.name, str(budget)))
    return issues


def validate_terminal_node(task_graph: TaskGraph) -> list[ValidationIssue]:
    """Validate at least one final node exists."""

    if any(node.kind == NodeKind.FINAL for node in task_graph.nodes):
        return []
    return [ValidationIssue("NO_TERMINAL_NODE", None, "Task graph has no final node")]


_V0_SUPPORTED_KINDS = {
    NodeKind.INTENT,
    NodeKind.TOOL,
    NodeKind.MODEL,
    NodeKind.COMPRESS,
    NodeKind.ROUTER,
    NodeKind.EXPAND,
    NodeKind.FINAL,
}


def validate_supported_node_kinds(task_graph: TaskGraph) -> list[ValidationIssue]:
    """Reject node kinds not implemented by the v0 executor."""

    issues = []
    for node in task_graph.nodes:
        if node.kind not in _V0_SUPPORTED_KINDS:
            issues.append(
                ValidationIssue(
                    "UNSUPPORTED_NODE_KIND",
                    node.name,
                    f'"{node.kind}" is not supported; use one of: '
                    + ", ".join(sorted(k.value for k in _V0_SUPPORTED_KINDS)),
                )
            )
    return issues


def validate_acyclic(task_graph: TaskGraph) -> list[ValidationIssue]:
    """Validate task edges form a DAG."""

    outgoing: dict[str, list[str]] = {node.id: [] for node in task_graph.nodes}
    for edge in task_graph.task_edges:
        outgoing.setdefault(edge.from_node_id, []).append(edge.to_node_id)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return False
        if node_id in visited:
            return True
        visiting.add(node_id)
        for next_id in outgoing.get(node_id, []):
            if not visit(next_id):
                return False
        visiting.remove(node_id)
        visited.add(node_id)
        return True

    for node_id in outgoing:
        if not visit(node_id):
            return [ValidationIssue("ILLEGAL_CYCLE", None, "Task graph contains a cycle")]
    return []


def validate_workspace_links(task_graph: TaskGraph, expected_goal_action: str | None = None) -> list[ValidationIssue]:
    """Validate minimal workspace link requirements."""

    issues: list[ValidationIssue] = []
    if expected_goal_action == "continue":
        has_followup = task_graph.metadata.get("parent_task_graph_id") or task_graph.metadata.get("has_workspace_link")
        if not has_followup:
            issues.append(ValidationIssue("MISSING_WORKSPACE_LINK", None, "continue requires a workspace link"))
    return issues


def validate_proposal_goal_action(
    proposal: TaskGraphProposal, expected_goal_action: str
) -> list[ValidationIssue]:
    """Validate planner echoed the authoritative goal action."""

    if proposal.goal_action.value != expected_goal_action:
        return [
            ValidationIssue(
                "GOAL_ACTION_MISMATCH",
                None,
                f"Expected {expected_goal_action}, got {proposal.goal_action.value}",
            )
        ]
    return []


def validate_workspace_edge_targets(task_graph: TaskGraph, edge_types: list[str]) -> list[ValidationIssue]:
    """Validate workspace edge type strings."""

    valid = {item.value for item in WorkspaceEdgeType}
    return [
        ValidationIssue("INVALID_WORKSPACE_TARGET", None, edge_type)
        for edge_type in edge_types
        if edge_type not in valid
    ]


def _port_schema(node: object, name: str, direction: str) -> str | None:
    for port in getattr(node, "ports", []):
        if port.name == name and port.direction == direction:
            return port.schema_ref
    return None


def _has_port(node: object, name: str, direction: str) -> bool:
    return _port_schema(node, name, direction) is not None


def validate_router_nodes(task_graph: TaskGraph) -> list[ValidationIssue]:
    """Validate router node output schema and outgoing edge branch labels."""

    issues: list[ValidationIssue] = []
    for node in task_graph.nodes:
        if node.kind != NodeKind.ROUTER:
            continue
        if node.output_schema != "RoutingDecision":
            issues.append(
                ValidationIssue(
                    "ROUTER_SCHEMA",
                    node.name,
                    f"Router output_schema must be 'RoutingDecision', got '{node.output_schema}'",
                )
            )
        outgoing = [edge for edge in task_graph.task_edges if edge.from_node_id == node.id]
        unlabelled = [edge for edge in outgoing if not edge.metadata.get("branch")]
        for edge in unlabelled:
            issues.append(
                ValidationIssue(
                    "ROUTER_UNLABELLED_BRANCH",
                    node.name,
                    f"Edge to node '{edge.to_node_id}' is missing metadata['branch']",
                )
            )
        distinct_branches = {edge.metadata.get("branch") for edge in outgoing if edge.metadata.get("branch")}
        if len(distinct_branches) < 2:
            issues.append(
                ValidationIssue(
                    "ROUTER_SINGLE_BRANCH",
                    node.name,
                    "Router has fewer than 2 distinct branch labels; branching has no effect",
                )
            )
    return issues
