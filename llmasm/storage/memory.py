"""In-memory storage backend."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from llmasm.errors import StorageError
from llmasm.graph.models import (
    Artifact,
    Checkpoint,
    Goal,
    MemoryItem,
    ModelCall,
    Run,
    RunNodeState,
    TaskEdge,
    TaskGraph,
    ToolCall,
    WorkspaceEdge,
    WorkspaceGraph,
    utcnow,
)
from llmasm.storage.base import ContextItem, FewShotExample


_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would could should "
    "may might shall can of in on at to for with by from into through about over after before "
    "and or but not no nor so yet both either neither as if that than then there this these "
    "those it its it's what which who whom when where how i you he she we they me him her us "
    "them my your his her our their mine yours his hers ours theirs what's that's here out up "
    "just also more very much such s t re ve ll d m".split()
)


def _meaningful_words(text: str) -> frozenset[str]:
    return frozenset(
        word.strip(".,!?;:'\"") for word in text.lower().split()
        if word.strip(".,!?;:'\"") and word.strip(".,!?;:'\"") not in _STOP_WORDS and len(word) > 1
    )


def _word_overlap_score(query: str, text: str) -> float:
    q = _meaningful_words(query)
    t = _meaningful_words(text)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


class InMemoryStorage:
    """Dictionary-backed storage for tests and local examples."""

    def __init__(self) -> None:
        self.workspace_graphs: dict[str, WorkspaceGraph] = {}
        self.task_graphs: dict[str, TaskGraph] = {}
        self.runs: dict[str, Run] = {}
        self.run_node_states: dict[tuple[str, str], RunNodeState] = {}
        self.task_edges: dict[str, TaskEdge] = {}
        self.workspace_edges: dict[str, WorkspaceEdge] = {}
        self.artifacts: dict[str, Artifact] = {}
        self.goals: dict[str, Goal] = {}
        self.checkpoints: dict[str, Checkpoint] = {}
        self.tool_calls: dict[str, ToolCall] = {}
        self.model_calls: dict[str, ModelCall] = {}
        self.memory_items: dict[str, MemoryItem] = {}
        self.compilation_failures: list[dict[str, object]] = []
        self._artifact_cache: dict[tuple[str, tuple[str, ...]], str] = {}
        self._few_shots: dict[str, list[FewShotExample]] = defaultdict(list)

    def create_workspace_graph(self, graph: WorkspaceGraph) -> None:
        self.workspace_graphs[graph.id] = deepcopy(graph)

    def load_workspace_graph(self, workspace_graph_id: str) -> WorkspaceGraph:
        return deepcopy(self._get(self.workspace_graphs, workspace_graph_id, "workspace graph"))

    def persist_task_graph(self, task_graph: TaskGraph) -> None:
        self.task_graphs[task_graph.id] = deepcopy(task_graph)
        for node in task_graph.nodes:
            for port in node.ports:
                if port.node_id is None:
                    port.node_id = node.id
        for edge in task_graph.task_edges:
            self.persist_task_edge(edge)
        proposal_json = task_graph.metadata.get("proposal_json")
        intent = str(task_graph.metadata.get("intent") or "")
        if proposal_json and intent:
            self._few_shots[task_graph.workspace_graph_id].append(
                FewShotExample(str(proposal_json), intent, task_graph.id)
            )

    def load_task_graph(self, task_graph_id: str) -> TaskGraph:
        return deepcopy(self._get(self.task_graphs, task_graph_id, "task graph"))

    def create_run(self, run: Run) -> None:
        self.runs[run.id] = deepcopy(run)

    def load_run(self, run_id: str) -> Run:
        return deepcopy(self._get(self.runs, run_id, "run"))

    def update_run(self, run: Run) -> None:
        self._get(self.runs, run.id, "run")
        self.runs[run.id] = deepcopy(run)

    def create_run_node_state(self, state: RunNodeState) -> None:
        self.run_node_states[(state.run_id, state.node_id)] = deepcopy(state)

    def list_run_node_states(self, run_id: str) -> list[RunNodeState]:
        states = [state for (rid, _), state in self.run_node_states.items() if rid == run_id]
        return deepcopy(sorted(states, key=lambda state: state.node_id))

    def update_run_node_state(self, state: RunNodeState) -> None:
        state.updated_at = utcnow()
        self.run_node_states[(state.run_id, state.node_id)] = deepcopy(state)

    def persist_task_edge(self, edge: TaskEdge) -> None:
        self.task_edges[edge.id] = deepcopy(edge)
        graph = self.task_graphs.get(edge.task_graph_id)
        if graph and all(existing.id != edge.id for existing in graph.task_edges):
            graph.task_edges.append(deepcopy(edge))

    def list_task_edges(self, task_graph_id: str) -> list[TaskEdge]:
        edges = [edge for edge in self.task_edges.values() if edge.task_graph_id == task_graph_id]
        return deepcopy(sorted(edges, key=lambda edge: edge.id))

    def persist_workspace_edge(self, edge: WorkspaceEdge) -> None:
        self.workspace_edges[edge.id] = deepcopy(edge)

    def list_workspace_edges(self, workspace_graph_id: str) -> list[WorkspaceEdge]:
        edges = [edge for edge in self.workspace_edges.values() if edge.workspace_graph_id == workspace_graph_id]
        return deepcopy(sorted(edges, key=lambda edge: edge.id))

    def load_workspace_edges_for_task(self, task_graph_id: str) -> list[WorkspaceEdge]:
        graph = self.load_task_graph(task_graph_id)
        ids = {task_graph_id, *(node.id for node in graph.nodes)}
        if graph.root_prompt_node_id:
            ids.add(graph.root_prompt_node_id)
        goal_id = graph.metadata.get("goal_id")
        if goal_id:
            ids.add(str(goal_id))
        edges = [
            edge
            for edge in self.workspace_edges.values()
            if edge.workspace_graph_id == graph.workspace_graph_id
            and (edge.from_id in ids or edge.to_id in ids)
        ]
        return deepcopy(sorted(edges, key=lambda edge: edge.id))

    def persist_artifact(self, artifact: Artifact) -> None:
        self.artifacts[artifact.id] = deepcopy(artifact)
        key = artifact.metadata.get("cache_key")
        inputs = artifact.metadata.get("input_artifact_ids")
        if key and isinstance(inputs, list) and artifact.superseded_by is None:
            self._artifact_cache[(str(key), tuple(sorted(map(str, inputs))))] = artifact.id

    def update_artifact(self, artifact_id: str, superseded_by: str) -> None:
        artifact = self._get(self.artifacts, artifact_id, "artifact")
        artifact.superseded_by = superseded_by
        cache_key = artifact.metadata.get("cache_key")
        inputs = artifact.metadata.get("input_artifact_ids")
        if cache_key and isinstance(inputs, list):
            self._artifact_cache.pop((str(cache_key), tuple(sorted(map(str, inputs)))), None)

    def load_artifact(self, artifact_id: str) -> Artifact:
        return deepcopy(self._get(self.artifacts, artifact_id, "artifact"))

    def list_artifacts(self, run_id: str | None = None) -> list[Artifact]:
        artifacts = self.artifacts.values()
        if run_id is not None:
            artifacts = [artifact for artifact in artifacts if artifact.run_id == run_id]
        return deepcopy(sorted(artifacts, key=lambda artifact: artifact.created_at))

    def persist_goal(self, goal: Goal) -> None:
        self.goals[goal.id] = deepcopy(goal)

    def load_active_goal(self, workspace_graph_id: str) -> Goal | None:
        active = [
            goal
            for goal in self.goals.values()
            if goal.workspace_graph_id == workspace_graph_id and goal.status == "active"
        ]
        if not active:
            return None
        return deepcopy(max(active, key=lambda goal: goal.updated_at))

    def load_goal(self, goal_id: str) -> Goal:
        return deepcopy(self._get(self.goals, goal_id, "goal"))

    def update_goal(self, goal: Goal) -> None:
        goal.updated_at = utcnow()
        self.goals[goal.id] = deepcopy(goal)

    def persist_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.checkpoints[checkpoint.id] = deepcopy(checkpoint)

    def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        items = [item for item in self.checkpoints.values() if item.run_id == run_id]
        return deepcopy(sorted(items, key=lambda item: item.created_at))

    def persist_tool_call(self, call: ToolCall) -> None:
        self.tool_calls[call.id] = deepcopy(call)

    def list_tool_calls(self, run_id: str) -> list[ToolCall]:
        calls = [call for call in self.tool_calls.values() if call.run_id == run_id]
        return deepcopy(sorted(calls, key=lambda call: call.created_at))

    def persist_model_call(self, call: ModelCall) -> None:
        self.model_calls[call.id] = deepcopy(call)

    def list_model_calls(self, run_id: str) -> list[ModelCall]:
        calls = [call for call in self.model_calls.values() if call.run_id == run_id]
        return deepcopy(sorted(calls, key=lambda call: call.created_at))

    def persist_memory_item(self, item: MemoryItem) -> None:
        self.memory_items[item.id] = deepcopy(item)

    def list_memory_items(self, workspace_graph_id: str) -> list[MemoryItem]:
        items = [item for item in self.memory_items.values() if item.workspace_graph_id == workspace_graph_id]
        return deepcopy(sorted(items, key=lambda item: item.created_at))

    def retrieve_few_shot_examples(
        self, workspace_graph_id: str, intent: str, limit: int
    ) -> list[FewShotExample]:
        examples = self._few_shots.get(workspace_graph_id, [])
        ranked = sorted(
            examples,
            key=lambda example: _word_overlap_score(intent, example.intent),
            reverse=True,
        )
        return deepcopy(ranked[:limit])

    def find_cached_artifact(
        self, node_execution_key: str, input_artifact_ids: list[str]
    ) -> Artifact | None:
        artifact_id = self._artifact_cache.get((node_execution_key, tuple(sorted(input_artifact_ids))))
        if artifact_id is None:
            return None
        artifact = self.artifacts[artifact_id]
        if artifact.superseded_by is not None:
            return None
        return deepcopy(artifact)

    def persist_compilation_failure(self, workspace_graph_id: str, payload: dict[str, object]) -> None:
        record = {"workspace_graph_id": workspace_graph_id, **payload}
        self.compilation_failures.append(deepcopy(record))

    def search_memory(
        self,
        workspace_graph_id: str,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 20,
        kinds: set[str] | None = None,
    ) -> list[MemoryItem]:
        filters = filters or {}
        items = self.list_memory_items(workspace_graph_id)
        if kinds is not None:
            items = [item for item in items if item.kind in kinds]
        for key, value in filters.items():
            if value is None:
                continue
            items = [item for item in items if item.metadata.get(key) == value]
        return sorted(items, key=lambda item: _word_overlap_score(query, item.text), reverse=True)[:limit]

    def retrieve_workspace_context(
        self,
        workspace_graph_id: str | list[str],
        query: str,
        budget_tokens: int,
        filters: dict[str, Any] | None = None,
        kinds: set[str] | None = None,
    ) -> list[ContextItem]:
        ids = [workspace_graph_id] if isinstance(workspace_graph_id, str) else workspace_graph_id
        all_items: list[MemoryItem] = []
        for ws_id in ids:
            all_items.extend(self.search_memory(ws_id, query, filters, limit=50, kinds=kinds))
        ranked = sorted(all_items, key=lambda item: _word_overlap_score(query, item.text), reverse=True)
        total = 0
        context: list[ContextItem] = []
        for item in ranked:
            score = _word_overlap_score(query, item.text)
            if score <= 0.0:
                continue
            tokens = max(1, len(item.text.split()))
            if total + tokens > budget_tokens:
                continue
            total += tokens
            context.append(
                ContextItem(
                    id=item.id,
                    kind="memory_item",
                    text=item.text,
                    score=score,
                    token_count=tokens,
                    item=item,
                )
            )
        return context

    @staticmethod
    def _get(mapping: dict[str, Any], key: str, label: str) -> Any:
        try:
            return mapping[key]
        except KeyError as exc:
            raise StorageError(f"Unknown {label}: {key}") from exc
