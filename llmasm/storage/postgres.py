"""PostgreSQL storage backend."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from llmasm.errors import StorageError
from llmasm.graph.models import (
    Artifact,
    Checkpoint,
    EmbeddingRef,
    Goal,
    MemoryItem,
    ModelCall,
    Node,
    Port,
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
from llmasm.storage.embeddings import ScoredMatch
from llmasm.storage.migrate import run_migrations


def _word_overlap_score(query: str, text: str) -> float:
    q = {word for word in query.lower().split() if word}
    t = {word for word in text.lower().split() if word}
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


class PostgresStorage:
    """psycopg3-backed storage for persistent sessions."""

    def __init__(self, dsn: str) -> None:
        try:
            self.conn = psycopg.connect(dsn, autocommit=True)
        except Exception as exc:
            raise StorageError(f"Cannot connect to Postgres: {exc}") from exc
        run_migrations(self.conn)

    def _cur(self):
        return self.conn.cursor(row_factory=dict_row)

    def _get_row(self, sql: str, params: tuple) -> dict:
        with self._cur() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        if row is None:
            raise StorageError(f"Row not found: {sql!r} params={params!r}")
        return row

    # ── WorkspaceGraph ────────────────────────────────────────────────────

    def create_workspace_graph(self, graph: WorkspaceGraph) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO workspace_graphs (id, name, status, metadata, created_at) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (graph.id, graph.name, graph.status, Jsonb(graph.metadata), graph.created_at),
            )

    def load_workspace_graph(self, workspace_graph_id: str) -> WorkspaceGraph:
        row = self._get_row("SELECT * FROM workspace_graphs WHERE id = %s", (workspace_graph_id,))
        return WorkspaceGraph(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
        )

    # ── TaskGraph ─────────────────────────────────────────────────────────

    def persist_task_graph(self, task_graph: TaskGraph) -> None:
        with self.conn.transaction():
            with self._cur() as cur:
                cur.execute(
                    "INSERT INTO task_graphs "
                    "(id, workspace_graph_id, root_prompt_node_id, parent_task_graph_id, "
                    "status, compiler_version, metadata, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (
                        task_graph.id,
                        task_graph.workspace_graph_id,
                        task_graph.root_prompt_node_id,
                        task_graph.parent_task_graph_id,
                        task_graph.status,
                        task_graph.compiler_version,
                        Jsonb(task_graph.metadata),
                        task_graph.created_at,
                    ),
                )
                for node in task_graph.nodes:
                    cur.execute(
                        "INSERT INTO nodes "
                        "(id, workspace_graph_id, task_graph_id, kind, name, "
                        "input_schema, output_schema, ports_json, execution_json, metadata, created_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        (
                            node.id,
                            node.workspace_graph_id,
                            node.task_graph_id,
                            node.kind,
                            node.name,
                            node.input_schema,
                            node.output_schema,
                            Jsonb([p.model_dump(mode="json") for p in node.ports]),
                            Jsonb(node.execution),
                            Jsonb(node.metadata),
                            node.created_at,
                        ),
                    )
                for edge in task_graph.task_edges:
                    cur.execute(
                        "INSERT INTO task_edges "
                        "(id, workspace_graph_id, task_graph_id, "
                        "from_node_id, from_port, to_node_id, to_port, transform, required, metadata) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        (
                            edge.id,
                            edge.workspace_graph_id,
                            edge.task_graph_id,
                            edge.from_node_id,
                            edge.from_port,
                            edge.to_node_id,
                            edge.to_port,
                            edge.transform,
                            edge.required,
                            Jsonb(edge.metadata),
                        ),
                    )

    def load_task_graph(self, task_graph_id: str) -> TaskGraph:
        row = self._get_row("SELECT * FROM task_graphs WHERE id = %s", (task_graph_id,))
        with self._cur() as cur:
            cur.execute(
                "SELECT * FROM nodes WHERE task_graph_id = %s ORDER BY id", (task_graph_id,)
            )
            node_rows = cur.fetchall()
            cur.execute(
                "SELECT * FROM task_edges WHERE task_graph_id = %s ORDER BY id", (task_graph_id,)
            )
            edge_rows = cur.fetchall()
        return TaskGraph(
            id=row["id"],
            workspace_graph_id=row["workspace_graph_id"],
            root_prompt_node_id=row["root_prompt_node_id"],
            parent_task_graph_id=row["parent_task_graph_id"],
            status=row["status"],
            compiler_version=row["compiler_version"],
            nodes=[self._row_to_node(r) for r in node_rows],
            task_edges=[self._row_to_task_edge(r) for r in edge_rows],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
        )

    def _row_to_node(self, row: dict) -> Node:
        return Node(
            id=row["id"],
            workspace_graph_id=row["workspace_graph_id"],
            task_graph_id=row["task_graph_id"],
            kind=row["kind"],
            name=row["name"],
            input_schema=row["input_schema"],
            output_schema=row["output_schema"],
            ports=[Port.model_validate(p) for p in (row["ports_json"] or [])],
            execution=row["execution_json"] or {},
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
        )

    def _row_to_task_edge(self, row: dict) -> TaskEdge:
        return TaskEdge(
            id=row["id"],
            workspace_graph_id=row["workspace_graph_id"],
            task_graph_id=row["task_graph_id"],
            from_node_id=row["from_node_id"],
            from_port=row["from_port"],
            to_node_id=row["to_node_id"],
            to_port=row["to_port"],
            transform=row["transform"],
            required=row["required"],
            metadata=row["metadata"] or {},
        )

    # ── Run ───────────────────────────────────────────────────────────────

    def create_run(self, run: Run) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO runs "
                "(id, workspace_graph_id, task_graph_id, status, "
                "program_counter_node_id, metadata, started_at, completed_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    run.id,
                    run.workspace_graph_id,
                    run.task_graph_id,
                    run.status,
                    run.program_counter_node_id,
                    Jsonb(run.metadata),
                    run.started_at,
                    run.completed_at,
                ),
            )

    def load_run(self, run_id: str) -> Run:
        row = self._get_row("SELECT * FROM runs WHERE id = %s", (run_id,))
        return Run(
            id=row["id"],
            workspace_graph_id=row["workspace_graph_id"],
            task_graph_id=row["task_graph_id"],
            status=row["status"],
            program_counter_node_id=row["program_counter_node_id"],
            metadata=row["metadata"] or {},
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    def update_run(self, run: Run) -> None:
        with self._cur() as cur:
            cur.execute(
                "UPDATE runs SET status = %s, program_counter_node_id = %s, "
                "metadata = %s, started_at = %s, completed_at = %s WHERE id = %s",
                (
                    run.status,
                    run.program_counter_node_id,
                    Jsonb(run.metadata),
                    run.started_at,
                    run.completed_at,
                    run.id,
                ),
            )

    # ── RunNodeState ──────────────────────────────────────────────────────

    def create_run_node_state(self, state: RunNodeState) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO run_node_states "
                "(run_id, node_id, status, attempts, last_error_json, "
                "output_artifact_ids, metadata, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    state.run_id,
                    state.node_id,
                    state.status,
                    state.attempts,
                    Jsonb(state.last_error) if state.last_error is not None else None,
                    Jsonb(state.output_artifact_ids),
                    Jsonb(state.metadata),
                    state.updated_at,
                ),
            )

    def list_run_node_states(self, run_id: str) -> list[RunNodeState]:
        with self._cur() as cur:
            cur.execute(
                "SELECT * FROM run_node_states WHERE run_id = %s ORDER BY node_id",
                (run_id,),
            )
            return [self._row_to_run_node_state(r) for r in cur.fetchall()]

    def update_run_node_state(self, state: RunNodeState) -> None:
        state.updated_at = utcnow()
        with self._cur() as cur:
            cur.execute(
                "UPDATE run_node_states SET status = %s, attempts = %s, "
                "last_error_json = %s, output_artifact_ids = %s, metadata = %s, updated_at = %s "
                "WHERE run_id = %s AND node_id = %s",
                (
                    state.status,
                    state.attempts,
                    Jsonb(state.last_error) if state.last_error is not None else None,
                    Jsonb(state.output_artifact_ids),
                    Jsonb(state.metadata),
                    state.updated_at,
                    state.run_id,
                    state.node_id,
                ),
            )

    def _row_to_run_node_state(self, row: dict) -> RunNodeState:
        return RunNodeState(
            run_id=row["run_id"],
            node_id=row["node_id"],
            status=row["status"],
            attempts=row["attempts"],
            last_error=row["last_error_json"],
            output_artifact_ids=row["output_artifact_ids"] or [],
            metadata=row["metadata"] or {},
            updated_at=row["updated_at"],
        )

    # ── TaskEdge ──────────────────────────────────────────────────────────

    def persist_task_edge(self, edge: TaskEdge) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO task_edges "
                "(id, workspace_graph_id, task_graph_id, "
                "from_node_id, from_port, to_node_id, to_port, transform, required, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    edge.id,
                    edge.workspace_graph_id,
                    edge.task_graph_id,
                    edge.from_node_id,
                    edge.from_port,
                    edge.to_node_id,
                    edge.to_port,
                    edge.transform,
                    edge.required,
                    Jsonb(edge.metadata),
                ),
            )

    def list_task_edges(self, task_graph_id: str) -> list[TaskEdge]:
        with self._cur() as cur:
            cur.execute(
                "SELECT * FROM task_edges WHERE task_graph_id = %s ORDER BY id",
                (task_graph_id,),
            )
            return [self._row_to_task_edge(r) for r in cur.fetchall()]

    # ── WorkspaceEdge ─────────────────────────────────────────────────────

    def persist_workspace_edge(self, edge: WorkspaceEdge) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO workspace_edges "
                "(id, workspace_graph_id, edge_type, "
                "from_type, from_id, to_type, to_id, "
                "from_port, to_port, reason, metadata, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    edge.id,
                    edge.workspace_graph_id,
                    edge.edge_type,
                    edge.from_type,
                    edge.from_id,
                    edge.to_type,
                    edge.to_id,
                    edge.from_port,
                    edge.to_port,
                    edge.reason,
                    Jsonb(edge.metadata),
                    edge.created_at,
                ),
            )

    def list_workspace_edges(self, workspace_graph_id: str) -> list[WorkspaceEdge]:
        with self._cur() as cur:
            cur.execute(
                "SELECT * FROM workspace_edges WHERE workspace_graph_id = %s ORDER BY id",
                (workspace_graph_id,),
            )
            return [self._row_to_workspace_edge(r) for r in cur.fetchall()]

    def load_workspace_edges_for_task(self, task_graph_id: str) -> list[WorkspaceEdge]:
        tg = self.load_task_graph(task_graph_id)
        ids = {task_graph_id, *(node.id for node in tg.nodes)}
        if tg.root_prompt_node_id:
            ids.add(tg.root_prompt_node_id)
        goal_id = tg.metadata.get("goal_id")
        if goal_id:
            ids.add(str(goal_id))
        ids_list = list(ids)
        with self._cur() as cur:
            cur.execute(
                "SELECT * FROM workspace_edges WHERE workspace_graph_id = %s "
                "AND (from_id = ANY(%s) OR to_id = ANY(%s)) ORDER BY id",
                (tg.workspace_graph_id, ids_list, ids_list),
            )
            return [self._row_to_workspace_edge(r) for r in cur.fetchall()]

    def _row_to_workspace_edge(self, row: dict) -> WorkspaceEdge:
        return WorkspaceEdge(
            id=row["id"],
            workspace_graph_id=row["workspace_graph_id"],
            edge_type=row["edge_type"],
            from_type=row["from_type"],
            from_id=row["from_id"],
            to_type=row["to_type"],
            to_id=row["to_id"],
            from_port=row["from_port"],
            to_port=row["to_port"],
            reason=row["reason"],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
        )

    # ── Artifact ──────────────────────────────────────────────────────────

    def persist_artifact(self, artifact: Artifact) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO artifacts "
                "(id, run_id, node_id, port, content_type, content_json, "
                "content_ref, token_count, superseded_by, metadata, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    artifact.id,
                    artifact.run_id,
                    artifact.node_id,
                    artifact.port,
                    artifact.content_type,
                    Jsonb(artifact.content_json) if artifact.content_json is not None else None,
                    artifact.content_ref,
                    artifact.token_count,
                    artifact.superseded_by,
                    Jsonb(artifact.metadata),
                    artifact.created_at,
                ),
            )

    def update_artifact(self, artifact_id: str, superseded_by: str) -> None:
        with self._cur() as cur:
            cur.execute(
                "UPDATE artifacts SET superseded_by = %s WHERE id = %s",
                (superseded_by, artifact_id),
            )

    def load_artifact(self, artifact_id: str) -> Artifact:
        row = self._get_row("SELECT * FROM artifacts WHERE id = %s", (artifact_id,))
        return self._row_to_artifact(row)

    def list_artifacts(self, run_id: str | None = None) -> list[Artifact]:
        with self._cur() as cur:
            if run_id is not None:
                cur.execute(
                    "SELECT * FROM artifacts WHERE run_id = %s ORDER BY created_at", (run_id,)
                )
            else:
                cur.execute("SELECT * FROM artifacts ORDER BY created_at")
            return [self._row_to_artifact(r) for r in cur.fetchall()]

    def _row_to_artifact(self, row: dict) -> Artifact:
        return Artifact(
            id=row["id"],
            run_id=row["run_id"],
            node_id=row["node_id"],
            port=row["port"],
            content_type=row["content_type"],
            content_json=row["content_json"],
            content_ref=row["content_ref"],
            token_count=row["token_count"],
            superseded_by=row["superseded_by"],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
        )

    # ── Goal ──────────────────────────────────────────────────────────────

    def persist_goal(self, goal: Goal) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO goals "
                "(id, workspace_graph_id, active_task_graph_id, text, status, "
                "metadata, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    goal.id,
                    goal.workspace_graph_id,
                    goal.active_task_graph_id,
                    goal.text,
                    goal.status,
                    Jsonb(goal.metadata),
                    goal.created_at,
                    goal.updated_at,
                ),
            )

    def load_active_goal(self, workspace_graph_id: str) -> Goal | None:
        with self._cur() as cur:
            cur.execute(
                "SELECT * FROM goals WHERE workspace_graph_id = %s AND status = 'active' "
                "ORDER BY updated_at DESC LIMIT 1",
                (workspace_graph_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_goal(row)

    def load_goal(self, goal_id: str) -> Goal:
        row = self._get_row("SELECT * FROM goals WHERE id = %s", (goal_id,))
        return self._row_to_goal(row)

    def update_goal(self, goal: Goal) -> None:
        goal.updated_at = utcnow()
        with self._cur() as cur:
            cur.execute(
                "UPDATE goals SET active_task_graph_id = %s, text = %s, status = %s, "
                "metadata = %s, updated_at = %s WHERE id = %s",
                (
                    goal.active_task_graph_id,
                    goal.text,
                    goal.status,
                    Jsonb(goal.metadata),
                    goal.updated_at,
                    goal.id,
                ),
            )

    def _row_to_goal(self, row: dict) -> Goal:
        return Goal(
            id=row["id"],
            workspace_graph_id=row["workspace_graph_id"],
            text=row["text"],
            status=row["status"],
            active_task_graph_id=row["active_task_graph_id"],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ── Checkpoint ────────────────────────────────────────────────────────

    def persist_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO checkpoints "
                "(id, run_id, program_counter_node_id, "
                "completed_node_ids, failed_node_ids, state_hash, state_json, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    checkpoint.id,
                    checkpoint.run_id,
                    checkpoint.program_counter_node_id,
                    Jsonb(checkpoint.completed_node_ids),
                    Jsonb(checkpoint.failed_node_ids),
                    checkpoint.state_hash,
                    Jsonb(checkpoint.state_json),
                    checkpoint.created_at,
                ),
            )

    def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        with self._cur() as cur:
            cur.execute(
                "SELECT * FROM checkpoints WHERE run_id = %s ORDER BY created_at", (run_id,)
            )
            return [
                Checkpoint(
                    id=r["id"],
                    run_id=r["run_id"],
                    program_counter_node_id=r["program_counter_node_id"],
                    completed_node_ids=r["completed_node_ids"] or [],
                    failed_node_ids=r["failed_node_ids"] or [],
                    state_hash=r["state_hash"],
                    state_json=r["state_json"] or {},
                    created_at=r["created_at"],
                )
                for r in cur.fetchall()
            ]

    # ── ToolCall ──────────────────────────────────────────────────────────

    def persist_tool_call(self, call: ToolCall) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO tool_calls "
                "(id, run_id, node_id, tool_name, input_json, "
                "output_artifact_id, status, latency_ms, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    call.id,
                    call.run_id,
                    call.node_id,
                    call.tool_name,
                    Jsonb(call.input_json) if call.input_json is not None else None,
                    call.output_artifact_id,
                    call.status,
                    call.latency_ms,
                    call.created_at,
                ),
            )

    def list_tool_calls(self, run_id: str) -> list[ToolCall]:
        with self._cur() as cur:
            cur.execute(
                "SELECT * FROM tool_calls WHERE run_id = %s ORDER BY created_at", (run_id,)
            )
            return [
                ToolCall(
                    id=r["id"],
                    run_id=r["run_id"],
                    node_id=r["node_id"],
                    tool_name=r["tool_name"],
                    input_json=r["input_json"],
                    output_artifact_id=r["output_artifact_id"],
                    status=r["status"],
                    latency_ms=r["latency_ms"],
                    created_at=r["created_at"],
                )
                for r in cur.fetchall()
            ]

    # ── ModelCall ─────────────────────────────────────────────────────────

    def persist_model_call(self, call: ModelCall) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO model_calls "
                "(id, run_id, node_id, provider, model, "
                "prompt_artifact_id, output_artifact_id, status, token_json, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    call.id,
                    call.run_id,
                    call.node_id,
                    call.provider,
                    call.model,
                    call.prompt_artifact_id,
                    call.output_artifact_id,
                    call.status,
                    Jsonb(call.token_json),
                    call.created_at,
                ),
            )

    def list_model_calls(self, run_id: str) -> list[ModelCall]:
        with self._cur() as cur:
            cur.execute(
                "SELECT * FROM model_calls WHERE run_id = %s ORDER BY created_at", (run_id,)
            )
            return [
                ModelCall(
                    id=r["id"],
                    run_id=r["run_id"],
                    node_id=r["node_id"],
                    provider=r["provider"],
                    model=r["model"],
                    prompt_artifact_id=r["prompt_artifact_id"],
                    output_artifact_id=r["output_artifact_id"],
                    status=r["status"],
                    token_json=r["token_json"] or {},
                    created_at=r["created_at"],
                )
                for r in cur.fetchall()
            ]

    # ── MemoryItem ────────────────────────────────────────────────────────

    def persist_memory_item(self, item: MemoryItem) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO memory_items "
                "(id, workspace_graph_id, kind, text, "
                "source_artifact_id, source_run_id, confidence, metadata, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    item.id,
                    item.workspace_graph_id,
                    item.kind,
                    item.text,
                    item.source_artifact_id,
                    item.source_run_id,
                    item.confidence,
                    Jsonb(item.metadata),
                    item.created_at,
                ),
            )

    def list_memory_items(self, workspace_graph_id: str) -> list[MemoryItem]:
        with self._cur() as cur:
            cur.execute(
                "SELECT * FROM memory_items WHERE workspace_graph_id = %s ORDER BY created_at",
                (workspace_graph_id,),
            )
            return [self._row_to_memory_item(r) for r in cur.fetchall()]

    def _row_to_memory_item(self, row: dict) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            workspace_graph_id=row["workspace_graph_id"],
            kind=row["kind"],
            text=row["text"],
            source_artifact_id=row["source_artifact_id"],
            source_run_id=row["source_run_id"],
            confidence=row["confidence"],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
        )

    # ── Few-shot examples and context search ──────────────────────────────

    def retrieve_few_shot_examples(
        self, workspace_graph_id: str, intent: str, limit: int
    ) -> list[FewShotExample]:
        """Rank prior successful proposals by word overlap against the current intent."""
        with self._cur() as cur:
            cur.execute(
                "SELECT id, metadata->>'intent' AS intent, "
                "metadata->>'proposal_json' AS proposal_json "
                "FROM task_graphs WHERE workspace_graph_id = %s "
                "AND metadata->>'proposal_json' IS NOT NULL "
                "AND metadata->>'intent' IS NOT NULL",
                (workspace_graph_id,),
            )
            rows = cur.fetchall()
        examples = [
            FewShotExample(
                proposal_json=r["proposal_json"],
                intent=r["intent"],
                task_graph_id=r["id"],
            )
            for r in rows
        ]
        ranked = sorted(
            examples,
            key=lambda e: _word_overlap_score(intent, e.intent),
            reverse=True,
        )
        return ranked[:limit]

    def find_cached_artifact(
        self, node_execution_key: str, input_artifact_ids: list[str]
    ) -> Artifact | None:
        with self._cur() as cur:
            cur.execute(
                "SELECT * FROM artifacts WHERE metadata->>'cache_key' = %s "
                "AND superseded_by IS NULL ORDER BY created_at DESC",
                (node_execution_key,),
            )
            rows = cur.fetchall()
        sorted_inputs = tuple(sorted(input_artifact_ids))
        for row in rows:
            inputs = row["metadata"].get("input_artifact_ids", [])
            if tuple(sorted(str(i) for i in inputs)) == sorted_inputs:
                return self._row_to_artifact(row)
        return None

    def persist_compilation_failure(
        self, workspace_graph_id: str, payload: dict[str, object]
    ) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO compilation_failures (workspace_graph_id, payload) VALUES (%s, %s)",
                (workspace_graph_id, Jsonb(payload)),
            )

    # ── Non-protocol bonus methods (used by compiler via hasattr) ─────────

    def search_memory(
        self,
        workspace_graph_id: str,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 20,
    ) -> list[MemoryItem]:
        filters = filters or {}
        items = self.list_memory_items(workspace_graph_id)
        for key, value in filters.items():
            if value is None:
                continue
            items = [item for item in items if item.metadata.get(key) == value]
        return sorted(
            items,
            key=lambda item: _word_overlap_score(query, item.text),
            reverse=True,
        )[:limit]

    def retrieve_workspace_context(
        self,
        workspace_graph_id: str | list[str],
        query: str,
        budget_tokens: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ContextItem]:
        ids = [workspace_graph_id] if isinstance(workspace_graph_id, str) else workspace_graph_id
        all_items: list[MemoryItem] = []
        for ws_id in ids:
            all_items.extend(self.search_memory(ws_id, query, filters, limit=50))
        ranked = sorted(all_items, key=lambda item: _word_overlap_score(query, item.text), reverse=True)
        total = 0
        context: list[ContextItem] = []
        for item in ranked:
            tokens = max(1, len(item.text.split()))
            if total + tokens > budget_tokens:
                continue
            total += tokens
            context.append(
                ContextItem(
                    id=item.id,
                    kind="memory_item",
                    text=item.text,
                    score=_word_overlap_score(query, item.text),
                    token_count=tokens,
                    item=item,
                )
            )
        return context


def _pg_cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity for Python fallback in PostgresEmbeddingStore."""
    from math import sqrt

    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    ln = sqrt(sum(a * a for a in left))
    rn = sqrt(sum(b * b for b in right))
    if ln == 0 or rn == 0:
        return 0.0
    return dot / (ln * rn)


class PostgresEmbeddingStore:
    """Postgres-backed embedding store.

    Uses pgvector ``ORDER BY <=>`` when the vector extension is present.  Falls
    back to loading all rows and computing cosine similarity in Python otherwise.

    Construct via :func:`PostgresEmbeddingStore.create` to auto-detect pgvector.
    The ``dimensions`` parameter must match the embedding model's output size.
    Changing dimensions after the column has been created raises :exc:`StorageError`.
    """

    def __init__(self, conn: Any, pgvector: bool = False, dimensions: int = 768) -> None:
        self.conn = conn
        self._pgvector = pgvector
        self._dimensions = dimensions
        if pgvector:
            self._ensure_vector_column()

    def _ensure_vector_column(self) -> None:
        """Create the vector column + index if absent; guard against dimension mismatches."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'embeddings'::regclass "
                "  AND attname = 'vector' "
                "  AND NOT attisdropped"
            )
            row = cur.fetchone()
        if row is None:
            # Column does not yet exist — create it with the requested dimensions.
            with self.conn.cursor() as cur:
                cur.execute(
                    f"ALTER TABLE embeddings "
                    f"ADD COLUMN IF NOT EXISTS vector vector({self._dimensions})"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_embeddings_vector "
                    "ON embeddings USING ivfflat (vector vector_cosine_ops) "
                    "WHERE vector IS NOT NULL"
                )
        else:
            existing = row[0]
            if existing != self._dimensions:
                raise StorageError(
                    f"embeddings.vector column has {existing} dimensions but "
                    f"RuntimeConfig.embedding_dimensions={self._dimensions}. "
                    "Drop the column and re-initialise to use a different model."
                )

    @classmethod
    def create(cls, conn: Any, dimensions: int = 768) -> "PostgresEmbeddingStore":
        """Return an instance with pgvector enabled when the extension is present."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
            )
            has_ext = cur.fetchone() is not None
        return cls(conn, pgvector=has_ext, dimensions=dimensions)

    # ── EmbeddingStore protocol ───────────────────────────────────────────

    def persist(self, ref: EmbeddingRef, vector: list[float]) -> None:
        with self.conn.cursor() as cur:
            if self._pgvector:
                vec_str = "[" + ",".join(str(v) for v in vector) + "]"
                cur.execute(
                    "INSERT INTO embeddings "
                    "(id, owner_type, owner_id, model, dimensions, text_hash, "
                    "vector_json, vector, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s) ON CONFLICT DO NOTHING",
                    (
                        ref.id, ref.owner_type, ref.owner_id, ref.model,
                        ref.dimensions, ref.text_hash,
                        Jsonb(vector), vec_str, ref.created_at,
                    ),
                )
            else:
                cur.execute(
                    "INSERT INTO embeddings "
                    "(id, owner_type, owner_id, model, dimensions, text_hash, "
                    "vector_json, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (
                        ref.id, ref.owner_type, ref.owner_id, ref.model,
                        ref.dimensions, ref.text_hash,
                        Jsonb(vector), ref.created_at,
                    ),
                )

    def search_similar(
        self,
        query_vector: list[float],
        filters: dict[str, object] | None,
        limit: int,
    ) -> list[ScoredMatch]:
        filters = filters or {}
        owner_type = str(filters["owner_type"]) if filters.get("owner_type") else None
        # Accept either workspace_graph_ids (list) or workspace_graph_id (single).
        _ws_ids = filters.get("workspace_graph_ids")
        _ws_id = filters.get("workspace_graph_id")
        workspace_ids: list[str] | None = None
        if _ws_ids:
            workspace_ids = list(_ws_ids)  # type: ignore[arg-type]
        elif _ws_id:
            workspace_ids = [str(_ws_id)]

        # When filtering by workspace, JOIN against the owning table rather than
        # post-filtering so the DB does the work and the LIMIT is accurate.
        join_sql = ""
        where_parts: list[str] = []
        params: list[object] = []
        if owner_type:
            where_parts.append("e.owner_type = %s")
            params.append(owner_type)
        if workspace_ids and owner_type == "memory_item":
            join_sql = "JOIN memory_items mi ON mi.id = e.owner_id"
            where_parts.append("mi.workspace_graph_id = ANY(%s)")
            params.append(workspace_ids)

        with self.conn.cursor(row_factory=dict_row) as cur:
            if self._pgvector:
                vec_str = "[" + ",".join(str(v) for v in query_vector) + "]"
                pg_where = where_parts + ["e.vector IS NOT NULL"]
                where_sql = "WHERE " + " AND ".join(pg_where)
                cur.execute(
                    f"SELECT e.*, 1 - (e.vector <=> %s::vector) AS cosine_score "
                    f"FROM embeddings e {join_sql} {where_sql} "
                    f"ORDER BY e.vector <=> %s::vector LIMIT %s",
                    (vec_str, *params, vec_str, limit),
                )
                rows = cur.fetchall()
                matches: list[ScoredMatch] = []
                for row in rows:
                    ref = self._row_to_ref(row)
                    item = self._load_item(ref.owner_type, ref.owner_id)
                    if item is not None:
                        score = float(row["cosine_score"]) if row.get("cosine_score") is not None else 0.0
                        matches.append(ScoredMatch(item=item, score=score, embedding_id=ref.id))
                return matches
            else:
                where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
                cur.execute(
                    f"SELECT e.* FROM embeddings e {join_sql} {where_sql} ORDER BY e.created_at",
                    params if params else [],
                )
                rows = cur.fetchall()

        scored: list[ScoredMatch] = []
        for row in rows:
            raw_vector = row.get("vector_json")
            if not raw_vector:
                continue
            score = _pg_cosine(query_vector, raw_vector)
            ref = self._row_to_ref(row)
            item = self._load_item(ref.owner_type, ref.owner_id)
            if item is not None:
                scored.append(ScoredMatch(item=item, score=score, embedding_id=ref.id))
        return sorted(scored, key=lambda m: m.score, reverse=True)[:limit]

    def find_by_owner(self, owner_type: str, owner_id: str) -> EmbeddingRef | None:
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM embeddings WHERE owner_type = %s AND owner_id = %s LIMIT 1",
                (owner_type, owner_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_ref(row)

    def has_embedding(self, owner_type: str, owner_id: str, text_hash: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM embeddings WHERE owner_type = %s AND owner_id = %s "
                "AND text_hash = %s LIMIT 1",
                (owner_type, owner_id, text_hash),
            )
            return cur.fetchone() is not None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_ref(row: dict) -> EmbeddingRef:
        return EmbeddingRef(
            id=row["id"],
            owner_type=row["owner_type"],
            owner_id=row["owner_id"],
            model=row["model"],
            dimensions=row["dimensions"],
            text_hash=row["text_hash"],
            created_at=row["created_at"],
        )

    def _load_item(self, owner_type: str, owner_id: str) -> MemoryItem | Artifact | None:
        """Fetch the owning MemoryItem or Artifact to attach to a ScoredMatch."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            if owner_type == "memory_item":
                cur.execute("SELECT * FROM memory_items WHERE id = %s", (owner_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return MemoryItem(
                    id=row["id"],
                    workspace_graph_id=row["workspace_graph_id"],
                    kind=row["kind"],
                    text=row["text"],
                    source_artifact_id=row["source_artifact_id"],
                    source_run_id=row["source_run_id"],
                    confidence=row["confidence"],
                    metadata=row["metadata"] or {},
                    created_at=row["created_at"],
                )
            if owner_type == "artifact":
                cur.execute("SELECT * FROM artifacts WHERE id = %s", (owner_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return Artifact(
                    id=row["id"],
                    run_id=row["run_id"],
                    node_id=row["node_id"],
                    port=row["port"],
                    content_type=row["content_type"],
                    content_json=row["content_json"],
                    content_ref=row["content_ref"],
                    token_count=row["token_count"],
                    superseded_by=row["superseded_by"],
                    metadata=row["metadata"] or {},
                    created_at=row["created_at"],
                )
        return None

