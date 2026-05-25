"""Graph visualization exporters."""

from __future__ import annotations

from typing import Any

from llmasm.analysis.run import RunAnalysis


def to_viewer_graph(analysis: RunAnalysis) -> dict[str, Any]:
    """Convert a run analysis into the first-party viewer JSON format."""

    states = {state.node_id: state for state in analysis.node_states}
    tool_calls = {call.node_id: call for call in analysis.tool_calls}
    model_calls = {call.node_id: call for call in analysis.model_calls}
    nodes: list[dict[str, Any]] = []
    for node in analysis.task_graph.nodes:
        state = states.get(node.id)
        call = tool_calls.get(node.id) or model_calls.get(node.id)
        metrics = {
            "attempts": state.attempts if state else 0,
            "artifacts": len(state.output_artifact_ids) if state else 0,
        }
        if call is not None and hasattr(call, "latency_ms"):
            metrics["latency_ms"] = getattr(call, "latency_ms")
        if call is not None and hasattr(call, "token_json"):
            token_json = getattr(call, "token_json")
            metrics["input_tokens"] = int(token_json.get("input_tokens", 0))
            metrics["output_tokens"] = int(token_json.get("output_tokens", 0))
        nodes.append(
            {
                "id": node.id,
                "label": node.name,
                "kind": node.kind.value,
                "status": state.status.value if state else "unknown",
                "subtitle": _node_subtitle(node.execution),
                "schema": {
                    "input": node.input_schema,
                    "output": node.output_schema,
                },
                "metrics": metrics,
                "metadata": node.metadata,
            }
        )
    edges = [
        {
            "id": edge.id,
            "source": edge.from_node_id,
            "target": edge.to_node_id,
            "label": f"{edge.from_port} -> {edge.to_port}",
            "type": "dataflow",
            "required": edge.required,
            "transform": edge.transform,
            "metadata": edge.metadata,
        }
        for edge in analysis.task_edges
    ]
    workspace_edges = [
        {
            "id": edge.id,
            "source": edge.from_id,
            "target": edge.to_id,
            "label": edge.edge_type.value,
            "type": "workspace",
            "reason": edge.reason,
            "metadata": edge.metadata,
        }
        for edge in analysis.workspace_edges
    ]
    return {
        "metadata": {
            "workspace_id": analysis.workspace.id,
            "workspace_name": analysis.workspace.name,
            "task_graph_id": analysis.task_graph.id,
            "run_id": analysis.run.id,
            "run_status": analysis.run.status.value,
            "token_usage": analysis.token_usage(),
        },
        "nodes": nodes,
        "edges": edges,
        "workspace_edges": workspace_edges,
    }


def to_mermaid(analysis: RunAnalysis) -> str:
    """Render the task graph as Mermaid flowchart text."""

    lines = ["flowchart TD"]
    for node in analysis.task_graph.nodes:
        label = _escape_mermaid(f"{node.kind.value}: {node.name}")
        lines.append(f"  {node.id}[\"{label}\"]")
    for edge in analysis.task_edges:
        label = _escape_mermaid(f"{edge.from_port} -> {edge.to_port}")
        lines.append(f"  {edge.from_node_id} -->|\"{label}\"| {edge.to_node_id}")
    return "\n".join(lines)


def to_dot(analysis: RunAnalysis) -> str:
    """Render the task graph as Graphviz DOT text."""

    lines = ["digraph llmasm {", "  rankdir=LR;"]
    for node in analysis.task_graph.nodes:
        label = _escape_dot(f"{node.kind.value}: {node.name}")
        lines.append(f'  "{node.id}" [label="{label}"];')
    for edge in analysis.task_edges:
        label = _escape_dot(f"{edge.from_port} -> {edge.to_port}")
        lines.append(f'  "{edge.from_node_id}" -> "{edge.to_node_id}" [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


def _node_subtitle(execution: dict[str, Any]) -> str:
    if "tool" in execution:
        return str(execution["tool"])
    if "model" in execution:
        return str(execution["model"])
    return ""


def _escape_mermaid(value: str) -> str:
    return value.replace('"', "'")


def _escape_dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
