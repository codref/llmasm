"""Synchronous task graph scheduler."""

from __future__ import annotations

from llmasm.graph.models import Node, NodeKind, NodeStatus, Run, TaskGraph
from llmasm.storage.base import Storage


class Scheduler:
    """Compute executable frontier for a run."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def next_node(self, run: Run) -> Node | None:
        """Return the next executable node or None."""

        graph = self.storage.load_task_graph(run.task_graph_id)
        states = {state.node_id: state for state in self.storage.list_run_node_states(run.id)}
        edges = self.storage.list_task_edges(run.task_graph_id)
        for node in graph.nodes:
            state = states.get(node.id)
            if state is None or state.status != NodeStatus.PENDING:
                continue
            incoming = [edge for edge in edges if edge.to_node_id == node.id]
            if self._blocked_by_not_found(run.id, node, graph):
                state.status = NodeStatus.SKIPPED
                state.metadata["skip_reason"] = "upstream_not_found"
                self.storage.update_run_node_state(state)
                continue
            complete = {NodeStatus.SUCCEEDED, NodeStatus.EXPANDED}
            if all(
                states.get(edge.from_node_id)
                and states[edge.from_node_id].status in complete
                for edge in incoming
            ):
                return node
        return None

    def terminal_succeeded(self, run: Run) -> bool:
        """Return true when a final node succeeded and no pending node is executable."""

        graph = self.storage.load_task_graph(run.task_graph_id)
        states = {state.node_id: state for state in self.storage.list_run_node_states(run.id)}
        return any(
            node.kind == NodeKind.FINAL
            and states.get(node.id)
            and states[node.id].status == NodeStatus.SUCCEEDED
            for node in graph.nodes
        )

    def _blocked_by_not_found(self, run_id: str, node: Node, graph: TaskGraph) -> bool:
        if node.kind != NodeKind.MODEL:
            return False
        states = {state.node_id: state for state in self.storage.list_run_node_states(run_id)}
        upstream_ids = {
            edge.from_node_id for edge in self.storage.list_task_edges(graph.id) if edge.to_node_id == node.id
        }
        return any(states.get(node_id) and states[node_id].metadata.get("not_found") for node_id in upstream_ids)
