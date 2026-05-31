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
        edges = self.storage.list_task_edges(run.task_graph_id)

        # Phase 1 — propagate SKIPPED status until stable.
        # Must complete before checking readiness so that downstream nodes see
        # correct upstream states regardless of node ordering in the graph.
        changed = True
        while changed:
            changed = False
            states = {state.node_id: state for state in self.storage.list_run_node_states(run.id)}
            for node in graph.nodes:
                state = states.get(node.id)
                if state is None or state.status != NodeStatus.PENDING:
                    continue
                incoming = [edge for edge in edges if edge.to_node_id == node.id]
                skip_reason: str | None = None
                if self._blocked_by_not_found(run.id, node, graph):
                    skip_reason = "upstream_not_found"
                elif self._blocked_by_router(run.id, node, graph, states) or self._all_upstream_skipped(node, incoming, states):
                    skip_reason = "router_branch_not_selected"
                if skip_reason:
                    state.status = NodeStatus.SKIPPED
                    state.metadata["skip_reason"] = skip_reason
                    self.storage.update_run_node_state(state)
                    changed = True

        # Phase 2 — find the first PENDING node whose every upstream has finished.
        states = {state.node_id: state for state in self.storage.list_run_node_states(run.id)}
        complete = {NodeStatus.SUCCEEDED, NodeStatus.EXPANDED, NodeStatus.SKIPPED}
        for node in graph.nodes:
            state = states.get(node.id)
            if state is None or state.status != NodeStatus.PENDING:
                continue
            incoming = [edge for edge in edges if edge.to_node_id == node.id]
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
        has_final = any(
            node.kind == NodeKind.FINAL
            and states.get(node.id)
            and states[node.id].status == NodeStatus.SUCCEEDED
            for node in graph.nodes
        )
        if has_final:
            return True
        # A run where every non-final node was skipped and a final succeeded via a router branch
        # is also terminal; handled above. No remaining case needed.
        return False

    def _blocked_by_not_found(self, run_id: str, node: Node, graph: TaskGraph) -> bool:
        if node.kind != NodeKind.MODEL:
            return False
        states = {state.node_id: state for state in self.storage.list_run_node_states(run_id)}
        upstream_ids = {
            edge.from_node_id for edge in self.storage.list_task_edges(graph.id) if edge.to_node_id == node.id
        }
        return any(states.get(node_id) and states[node_id].metadata.get("not_found") for node_id in upstream_ids)

    def _blocked_by_router(
        self,
        run_id: str,
        node: Node,
        graph: TaskGraph,
        states: dict,
    ) -> bool:
        """Return True if this node is on a router branch that was not selected."""
        nodes_by_id = {n.id: n for n in graph.nodes}
        for edge in self.storage.list_task_edges(graph.id):
            if edge.to_node_id != node.id:
                continue
            src = nodes_by_id.get(edge.from_node_id)
            if src is None or src.kind != NodeKind.ROUTER:
                continue
            router_state = states.get(src.id)
            if router_state is None or router_state.status != NodeStatus.SUCCEEDED:
                continue
            selected = router_state.metadata.get("selected_branch")
            edge_branch = edge.metadata.get("branch")
            if edge_branch is not None and selected is not None and edge_branch != selected:
                return True
        return False

    def _all_upstream_skipped(self, node: Node, incoming: list, states: dict) -> bool:
        """Return True when every upstream node was skipped (transitive branch pruning)."""
        if not incoming:
            return False
        return all(
            states.get(edge.from_node_id) and states[edge.from_node_id].status == NodeStatus.SKIPPED
            for edge in incoming
        )
