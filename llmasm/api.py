"""Thin public facade."""

from __future__ import annotations

from typing import Any

from llmasm.analysis.run import RunAnalysis, query_run
from llmasm.compiler.compiler import Compiler
from llmasm.config import RuntimeConfig
from llmasm.graph.models import Run, WorkspaceGraph
from llmasm.graph.registry import SchemaRegistry, default_schema_registry
from llmasm.graph.transforms import TransformRegistry, default_transform_registry
from llmasm.ids import new_id
from llmasm.conversation.chat import chat_turn
from llmasm.conversation.ingestion import maybe_ingest_long_source
from llmasm.conversation.memory import store_source_summary
from llmasm.providers.base import LLMProvider
from llmasm.runtime.executor import Executor
from llmasm.schemas import FinalAnswer, Summary
from llmasm.storage.base import Storage
from llmasm.storage.embeddings import EmbeddingStore, NullEmbeddingStore
from llmasm.tools.registry import ToolRegistry


class LLMASM:
    """Ergonomic facade over compiler, executor, and analysis."""

    def __init__(
        self,
        *,
        storage: Storage,
        provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        runtime_config: RuntimeConfig | None = None,
        schema_registry: SchemaRegistry | None = None,
        transform_registry: TransformRegistry | None = None,
        embedding_store: EmbeddingStore | None = None,
    ) -> None:
        self.storage = storage
        self.provider = provider
        self.runtime_config = runtime_config or RuntimeConfig()
        self.schema_registry = schema_registry or default_schema_registry()
        self.transform_registry = transform_registry or default_transform_registry()
        self.tool_registry = tool_registry or ToolRegistry(self.schema_registry)
        self.embedding_store = embedding_store

    def create_workspace(self, name: str, **metadata: object) -> str:
        """Create a workspace graph."""

        workspace = WorkspaceGraph(id=new_id("workspace"), name=name, metadata=dict(metadata))
        self.storage.create_workspace_graph(workspace)
        return workspace.id

    def compile(self, workspace_id: str, prompt: str) -> str:
        """Compile a prompt into a task graph ID."""

        compiler = Compiler(
            storage=self.storage,
            planner=self.provider,
            schema_registry=self.schema_registry,
            transform_registry=self.transform_registry,
            tool_registry=self.tool_registry,
            runtime_config=self.runtime_config,
        )
        return compiler.compile_into_workspace(workspace_id, prompt)

    def run(self, task_graph_id: str) -> str:
        """Create and execute a run for a task graph."""

        graph = self.storage.load_task_graph(task_graph_id)
        run = Run(
            id=new_id("run"),
            workspace_graph_id=graph.workspace_graph_id,
            task_graph_id=task_graph_id,
        )
        self.storage.create_run(run)
        executor = Executor(
            storage=self.storage,
            tool_registry=self.tool_registry,
            provider=self.provider,
            schema_registry=self.schema_registry,
            transform_registry=self.transform_registry,
            runtime_config=self.runtime_config,
            embedding_store=self.embedding_store,
        )
        executor.execute(run.id)
        return run.id

    def ask(
        self,
        workspace_id: str,
        prompt: str,
        *,
        out_info: dict[str, Any] | None = None,
    ) -> FinalAnswer:
        """Compile and execute a prompt, returning the final answer artifact.

        Args:
            out_info: Optional dict that will be populated with metadata about
                the turn, including ``run_id``, ``task_graph_id``, and
                ``chunked_source``.
        """

        ingestion = maybe_ingest_long_source(
            workspace_id,
            prompt,
            storage=self.storage,
            provider=self.provider,
            runtime_config=self.runtime_config,
            embedding_store=self.embedding_store,
        )
        task_graph_id = self.compile(workspace_id, ingestion.effective_prompt)
        run_id = self.run(task_graph_id)
        graph = self.storage.load_task_graph(task_graph_id)
        final_node_ids = {node.id for node in graph.nodes if node.kind == "final"}
        artifacts = [
            artifact
            for artifact in self.storage.list_artifacts(run_id)
            if artifact.node_id in final_node_ids
        ]

        # Persist any planner-emitted summary artifacts as workspace memory.
        if ingestion.source_id is not None:
            self._persist_summary_artifacts(run_id, ingestion.source_id)

        if out_info is not None:
            out_info["run_id"] = run_id
            out_info["task_graph_id"] = task_graph_id
            out_info["chunked_source"] = ingestion.source_id is not None

        if not artifacts:
            return FinalAnswer(text="", sources=[])
        return FinalAnswer.model_validate(artifacts[-1].content_json)

    def _persist_summary_artifacts(self, run_id: str, source_id: str) -> None:
        """Store Summary artifacts produced by summary nodes as workspace memory."""

        graph = self.storage.load_task_graph(self.storage.load_run(run_id).task_graph_id)
        summary_node_ids = {
            node.id
            for node in graph.nodes
            if node.kind == "model" and node.metadata.get("is_summary_node")
        }
        for artifact in self.storage.list_artifacts(run_id):
            if artifact.node_id not in summary_node_ids or artifact.port != "output":
                continue
            try:
                summary_text = Summary.model_validate(artifact.content_json).text
            except Exception:
                continue
            store_source_summary(
                graph.workspace_graph_id,
                summary_text,
                source_id=source_id,
                storage=self.storage,
                runtime_config=self.runtime_config,
                provider=self.provider,
                embedding_store=self.embedding_store or NullEmbeddingStore(),
                source_run_id=run_id,
            )

    def chat(
        self,
        workspace_id: str,
        prompt: str,
        *,
        out_info: dict[str, Any] | None = None,
        turn: int | None = None,
    ) -> FinalAnswer:
        """Conversation fast path: compile and execute without planner.

        Creates a deterministic ``intent -> model -> final`` graph, applies
        strict grounded-QA rules when source passages exist, and stores
        structured conversation memory.

        Args:
            out_info: Optional dict that will be populated with metadata about
                the turn, including ``instruction_tokens`` and ``run_id``.
            turn: Optional turn number for structured memory tracking.
        """
        return chat_turn(
            workspace_graph_id=workspace_id,
            prompt=prompt,
            storage=self.storage,
            provider=self.provider,
            runtime_config=self.runtime_config,
            tool_registry=self.tool_registry,
            schema_registry=self.schema_registry,
            transform_registry=self.transform_registry,
            embedding_store=self.embedding_store,
            out_info=out_info,
            turn=turn,
        )

    def query_run(self, run_id: str) -> RunAnalysis:
        """Return a materialized run analysis."""

        return query_run(self.storage, run_id)
