"""Single-process synchronous executor."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from llmasm.config import RuntimeConfig
from llmasm.errors import ExecutionError, FatalError, RetryableError
from llmasm.graph.models import (
    Artifact,
    Checkpoint,
    ModelCall,
    Node,
    NodeKind,
    NodeStatus,
    Run,
    RunNodeState,
    RunStatus,
    ToolCall,
)
from llmasm.graph.registry import SchemaRegistry
from llmasm.graph.transforms import TransformRegistry
from llmasm.ids import new_id
from llmasm.providers.base import LLMProvider
from llmasm.runtime.context import select_context
from llmasm.runtime.expansion import ExpansionRequest, apply_expansion
from llmasm.runtime.scheduler import Scheduler
from llmasm.schemas import FinalAnswer, NotFound, RawText, Summary
from llmasm.storage.base import ContextItem, Storage
from llmasm.storage.embeddings import EmbeddingStore, NullEmbeddingStore, embed_and_persist
from llmasm.tools.registry import ToolRegistry


class Executor:
    """Execute persisted task graphs."""

    def __init__(
        self,
        *,
        storage: Storage,
        tool_registry: ToolRegistry,
        provider: LLMProvider,
        schema_registry: SchemaRegistry,
        transform_registry: TransformRegistry,
        runtime_config: RuntimeConfig,
        embedding_store: EmbeddingStore | None = None,
    ) -> None:
        self.storage = storage
        self.tool_registry = tool_registry
        self.provider = provider
        self.schema_registry = schema_registry
        self.transform_registry = transform_registry
        self.runtime_config = runtime_config
        self.embedding_store = embedding_store or NullEmbeddingStore()
        self.scheduler = Scheduler(storage)

    def execute(self, run_id: str) -> Run:
        """Execute a run to completion or failure."""

        run = self.storage.load_run(run_id)
        graph = self.storage.load_task_graph(run.task_graph_id)
        self._ensure_node_states(run, graph.nodes)
        run.status = RunStatus.RUNNING
        run.started_at = run.started_at or datetime.now(UTC)
        self.storage.update_run(run)
        while True:
            node = self.scheduler.next_node(run)
            if node is None:
                if self.scheduler.terminal_succeeded(run):
                    run.status = RunStatus.SUCCEEDED
                    run.completed_at = datetime.now(UTC)
                    self.storage.update_run(run)
                    return run
                pending = [
                    state for state in self.storage.list_run_node_states(run.id)
                    if state.status == NodeStatus.PENDING
                ]
                if pending:
                    run.status = RunStatus.FAILED
                    run.completed_at = datetime.now(UTC)
                    self.storage.update_run(run)
                    raise ExecutionError("Run stalled with pending nodes")
                run.status = RunStatus.SUCCEEDED
                run.completed_at = datetime.now(UTC)
                self.storage.update_run(run)
                return run
            run.program_counter_node_id = node.id
            self.storage.update_run(run)
            self._checkpoint(run)
            try:
                self._execute_node(run, node)
            except FatalError:
                self._mark_failed(run, node, "fatal")
                raise
            except RetryableError as exc:
                self._mark_retryable(run, node, str(exc))
                raise
            except Exception as exc:
                self._mark_failed(run, node, str(exc))
                run.status = RunStatus.FAILED
                run.completed_at = datetime.now(UTC)
                self.storage.update_run(run)
                raise ExecutionError(f"Node {node.name} failed: {exc}") from exc
            self._checkpoint(run)

    def _execute_node(self, run: Run, node: Node) -> None:
        state = self._state(run.id, node.id)
        state.status = NodeStatus.RUNNING
        state.attempts += 1
        self.storage.update_run_node_state(state)
        direct_inputs, input_artifact_ids = self._gather_inputs(run, node)
        allow_cache = self._allow_cache(node)
        cache_key = self._cache_key(node)
        if allow_cache:
            cached = self.storage.find_cached_artifact(cache_key, input_artifact_ids)
            if cached is not None:
                artifact = cached.model_copy(update={"id": new_id("artifact"), "run_id": run.id})
                artifact.metadata["cached_from"] = cached.id
                self.storage.persist_artifact(artifact)
                self._mark_succeeded(run, node, [artifact.id], {"cache_hit": True})
                return
        output = self._invoke(run, node, direct_inputs)
        output_port = node.output_port_name()
        artifact = Artifact(
            id=new_id("artifact"),
            run_id=run.id,
            node_id=node.id,
            port=output_port,
            content_json=output.model_dump(mode="json"),
            token_count=self.runtime_config.tokenizer.count_tokens(json.dumps(output.model_dump())),
            metadata={"cache_key": cache_key, "input_artifact_ids": input_artifact_ids},
        )
        self.storage.persist_artifact(artifact)
        if node.kind in {NodeKind.MODEL, NodeKind.COMPRESS}:
            text = getattr(output, "text", json.dumps(output.model_dump()))
            embed_and_persist(
                text,
                "artifact",
                artifact.id,
                self.runtime_config,
                self.provider,
                self.embedding_store,
            )
        metadata: dict[str, Any] = {}
        if isinstance(output, NotFound):
            metadata["not_found"] = True
        self._mark_succeeded(run, node, [artifact.id], metadata)

    def _invoke(self, run: Run, node: Node, direct_inputs: dict[str, BaseModel]) -> BaseModel:
        if node.kind == NodeKind.INTENT:
            payload = node.execution.get("output") or node.metadata.get("output") or node.metadata
            if node.output_schema == "RawText":
                return RawText(text=self._intent_raw_text(payload, node.metadata))
            return self._model_for_schema(node.output_schema).model_validate(payload)
        if node.kind == NodeKind.TOOL:
            tool_name = str(node.execution.get("tool"))
            tool = self.tool_registry.get(tool_name)
            if not direct_inputs:
                raise ExecutionError(
                    f"Tool node {node.name} has no input artifact; connect an upstream node to its input port"
                )
            input_value = self._single_or_model(node.input_schema, direct_inputs)
            start = perf_counter()
            output = tool.invoke(input_value)
            elapsed = int((perf_counter() - start) * 1000)
            self.storage.persist_tool_call(
                ToolCall(
                    id=f"toolcall_{uuid4().hex}",
                    run_id=run.id,
                    node_id=node.id,
                    tool_name=tool_name,
                    input_json=input_value.model_dump(mode="json"),
                    status="succeeded",
                    latency_ms=elapsed,
                )
            )
            return output
        if node.kind in {NodeKind.MODEL, NodeKind.COMPRESS}:
            selected = select_context(
                storage=self.storage,
                runtime_config=self.runtime_config,
                run=run,
                node=node,
                direct_inputs=direct_inputs,
                embedding_store=self.embedding_store,
            )
            prompt = self._render_node_prompt(node, selected.direct_inputs, selected.items)
            prompt_artifact = Artifact(
                id=new_id("artifact"),
                run_id=run.id,
                node_id=node.id,
                port="prompt",
                content_json={"text": prompt},
                token_count=self.runtime_config.tokenizer.count_tokens(prompt),
            )
            self.storage.persist_artifact(prompt_artifact)
            result = self.provider.generate(
                prompt,
                {"model": node.execution.get("model") or self.runtime_config.default_model},
                None,
            )
            output = self._coerce_model_output(node.output_schema, result.text)
            self.storage.persist_model_call(
                ModelCall(
                    id=f"modelcall_{uuid4().hex}",
                    run_id=run.id,
                    node_id=node.id,
                    provider=getattr(self.provider, "name", "provider"),
                    model=str(node.execution.get("model") or self.runtime_config.default_model),
                    prompt_artifact_id=prompt_artifact.id,
                    status="succeeded",
                    token_json=result.token_usage or {},
                )
            )
            return output
        if node.kind == NodeKind.FINAL:
            input_value = next(iter(direct_inputs.values()), RawText(text=""))
            if isinstance(input_value, FinalAnswer):
                return input_value
            if isinstance(input_value, NotFound):
                return FinalAnswer(text=input_value.detail, sources=[])
            text = getattr(input_value, "text", json.dumps(input_value.model_dump()))
            sources = []
            source_id = getattr(input_value, "source_id", None)
            if source_id:
                sources.append(source_id)
            return FinalAnswer(text=str(text), sources=sources)
        if node.kind == NodeKind.EXPAND:
            request_payload = node.execution.get("expansion_request") or node.metadata.get("expansion_request")
            request = ExpansionRequest.model_validate(request_payload)
            expansion = apply_expansion(run=run, source_node=node, request=request, storage=self.storage)
            state = self._state(run.id, node.id)
            state.status = NodeStatus.EXPANDED
            state.metadata["expansion"] = expansion.model_dump()
            self.storage.update_run_node_state(state)
            return RawText(text=json.dumps(expansion.model_dump(), sort_keys=True))
        if node.kind in {NodeKind.MEMORY_QUERY, NodeKind.ROUTER, NodeKind.GOAL, NodeKind.OBSERVATION}:
            raise ExecutionError(f"Unsupported node kind in v0: {node.kind.value}")
        raise ExecutionError(f"Unsupported node kind in v0: {node.kind.value}")

    def _gather_inputs(self, run: Run, node: Node) -> tuple[dict[str, BaseModel], list[str]]:
        graph = self.storage.load_task_graph(run.task_graph_id)
        edges = [edge for edge in self.storage.list_task_edges(graph.id) if edge.to_node_id == node.id]
        direct: dict[str, BaseModel] = {}
        input_ids: list[str] = []
        states = {state.node_id: state for state in self.storage.list_run_node_states(run.id)}
        for edge in edges:
            state = states.get(edge.from_node_id)
            if not state or not state.output_artifact_ids:
                if edge.required:
                    raise ExecutionError(f"Missing required input {edge.to_port}")
                continue
            artifacts = [self.storage.load_artifact(artifact_id) for artifact_id in state.output_artifact_ids]
            artifact = next((item for item in artifacts if item.port == edge.from_port), artifacts[0])
            input_ids.append(
                sha256(
                    json.dumps(artifact.content_json, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
            )
            from_node = next(item for item in graph.nodes if item.id == edge.from_node_id)
            value = self._artifact_to_model(from_node.output_schema, artifact.content_json)
            if edge.transform:
                value = self.transform_registry.apply(edge.transform, value, edge.metadata)
            direct[edge.to_port] = value
        return direct, input_ids

    def _artifact_to_model(self, schema_ref: str | None, content: Any) -> BaseModel:
        model = self._model_for_schema(schema_ref)
        if isinstance(content, dict):
            return model.model_validate(content)
        return model.model_validate({"text": str(content)})

    def _model_for_schema(self, schema_ref: str | None) -> type[BaseModel]:
        if schema_ref is None:
            return RawText
        return self.schema_registry.get(schema_ref)

    def _single_or_model(self, schema_ref: str | None, direct_inputs: dict[str, BaseModel]) -> BaseModel:
        if len(direct_inputs) == 1:
            return next(iter(direct_inputs.values()))
        model = self._model_for_schema(schema_ref)
        data = {key: value.model_dump(mode="json") for key, value in direct_inputs.items()}
        return model.model_validate(data)

    def _coerce_model_output(self, schema_ref: str | None, text: str) -> BaseModel:
        model = self._model_for_schema(schema_ref)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return model.model_validate(parsed)
        except json.JSONDecodeError:
            pass
        if model is Summary:
            return Summary(text=text)
        if model is FinalAnswer:
            return FinalAnswer(text=text, sources=[])
        if model is RawText:
            return RawText(text=text)
        return model.model_validate({"text": text})

    def _intent_raw_text(self, payload: Any, metadata: dict[str, Any]) -> str:
        if isinstance(payload, dict):
            value = (
                payload.get("prompt")
                or payload.get("text")
                or payload.get("output.text")
                or metadata.get("prompt")
                or metadata.get("text")
                or metadata.get("output.text")
            )
            return str(value if value is not None else payload)
        if isinstance(payload, str) and payload in {"text", "prompt", "output.text"}:
            value = metadata.get(payload) or metadata.get("text") or metadata.get("prompt")
            return str(value if value is not None else payload)
        return str(payload)

    def _render_node_prompt(
        self,
        node: Node,
        direct_inputs: dict[str, BaseModel],
        context_items: list[ContextItem] | None = None,
    ) -> str:
        instruction = (
            node.metadata.get("instruction")
            or node.execution.get("prompt_template")
            or node.metadata.get("description")
            or node.name
        )
        if context_items:
            prior = "\n".join(item.text for item in context_items)
            instruction = (
                f"{instruction}\n\n"
                f"--- Prior conversation ---\n{prior}\n---\n\n"
                f"Use the prior conversation above to resolve references and understand "
                f"what the question is about. Answer from your own knowledge."
            )
        payload: dict[str, Any] = {
            "instruction": instruction,
            "inputs": {name: value.model_dump(mode="json") for name, value in sorted(direct_inputs.items())},
        }
        return json.dumps(payload, sort_keys=True)

    def _ensure_node_states(self, run: Run, nodes: list[Node]) -> None:
        existing = {state.node_id for state in self.storage.list_run_node_states(run.id)}
        for node in nodes:
            if node.id not in existing:
                self.storage.create_run_node_state(RunNodeState(run_id=run.id, node_id=node.id))

    def _state(self, run_id: str, node_id: str) -> RunNodeState:
        for state in self.storage.list_run_node_states(run_id):
            if state.node_id == node_id:
                return state
        raise ExecutionError(f"Missing run node state for {node_id}")

    def _mark_succeeded(
        self, run: Run, node: Node, artifact_ids: list[str], metadata: dict[str, Any] | None = None
    ) -> None:
        state = self._state(run.id, node.id)
        if state.status != NodeStatus.EXPANDED:
            state.status = NodeStatus.SUCCEEDED
        state.output_artifact_ids.extend(artifact_ids)
        state.metadata.update(metadata or {})
        self.storage.update_run_node_state(state)

    def _mark_failed(self, run: Run, node: Node, error: str) -> None:
        state = self._state(run.id, node.id)
        state.status = NodeStatus.FAILED
        state.last_error = {"message": error}
        self.storage.update_run_node_state(state)

    def _mark_retryable(self, run: Run, node: Node, error: str) -> None:
        state = self._state(run.id, node.id)
        state.status = NodeStatus.RETRYABLE
        state.last_error = {"message": error}
        self.storage.update_run_node_state(state)

    def _checkpoint(self, run: Run) -> None:
        states = self.storage.list_run_node_states(run.id)
        completed = [state.node_id for state in states if state.status == NodeStatus.SUCCEEDED]
        failed = [state.node_id for state in states if state.status == NodeStatus.FAILED]
        payload = {
            "program_counter_node_id": run.program_counter_node_id,
            "completed_node_ids": completed,
            "failed_node_ids": failed,
        }
        self.storage.persist_checkpoint(
            Checkpoint(
                id=new_id("checkpoint"),
                run_id=run.id,
                program_counter_node_id=run.program_counter_node_id,
                completed_node_ids=completed,
                failed_node_ids=failed,
                state_hash=sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest(),
                state_json=payload,
            )
        )

    @staticmethod
    def _allow_cache(node: Node) -> bool:
        if "allow_cache" in node.execution:
            return bool(node.execution["allow_cache"])
        return node.kind == NodeKind.TOOL

    @staticmethod
    def _cache_key(node: Node) -> str:
        return json.dumps(
            {"kind": node.kind.value, "name": node.name, "execution": node.execution},
            sort_keys=True,
        )
