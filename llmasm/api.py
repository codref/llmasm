"""Thin public facade."""

from __future__ import annotations

from llmasm.analysis.run import RunAnalysis, query_run
from llmasm.compiler.compiler import Compiler
from llmasm.config import RuntimeConfig
from llmasm.graph.models import Run, WorkspaceGraph
from llmasm.graph.registry import SchemaRegistry, default_schema_registry
from llmasm.graph.transforms import TransformRegistry, default_transform_registry
from llmasm.ids import new_id
from llmasm.providers.base import LLMProvider
from llmasm.runtime.executor import Executor
from llmasm.conversation.chat import chat_turn
from llmasm.schemas import FinalAnswer
from llmasm.storage.base import Storage
from llmasm.storage.embeddings import EmbeddingStore
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

    def ask(self, workspace_id: str, prompt: str) -> FinalAnswer:
        """Compile and execute a prompt, returning the final answer artifact."""

        task_graph_id = self.compile(workspace_id, prompt)
        run_id = self.run(task_graph_id)
        graph = self.storage.load_task_graph(task_graph_id)
        final_node_ids = {node.id for node in graph.nodes if node.kind == "final"}
        artifacts = [
            artifact
            for artifact in self.storage.list_artifacts(run_id)
            if artifact.node_id in final_node_ids
        ]
        if not artifacts:
            return FinalAnswer(text="", sources=[])
        return FinalAnswer.model_validate(artifacts[-1].content_json)

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
