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
from llmasm.providers.base import LLMProvider, ModelOutput
from llmasm.runtime.context import select_context
from llmasm.runtime.expansion import ExpansionRequest, apply_expansion
from llmasm.runtime.scheduler import Scheduler
from llmasm.schemas import FinalAnswer, NotFound, RawText, RoutingDecision, Summary, ToolCallRequest, ToolCallResult
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
                    graph = self.storage.load_task_graph(run.task_graph_id)
                    nodes_by_id = {n.id: n for n in graph.nodes}
                    all_states = self.storage.list_run_node_states(run.id)
                    state_summary = "; ".join(
                        f"{nodes_by_id[s.node_id].name}({nodes_by_id[s.node_id].kind})={s.status}"
                        for s in all_states
                        if s.node_id in nodes_by_id
                    )
                    raise ExecutionError(f"Run stalled with pending nodes. Node states: {state_summary}")
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
            token_count=self.runtime_config.tokenizer.count_tokens(json.dumps(output.model_dump(mode="json"))),
            metadata={"cache_key": cache_key, "input_artifact_ids": input_artifact_ids},
        )
        self.storage.persist_artifact(artifact)
        if node.kind in {NodeKind.MODEL, NodeKind.COMPRESS}:
            text = getattr(output, "text", json.dumps(output.model_dump(mode="json")))
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
        if isinstance(output, RoutingDecision):
            metadata["selected_branch"] = output.selected_branch
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
            return self._invoke_model_node(run, node, direct_inputs)

        if node.kind == NodeKind.FINAL:
            input_value = next(iter(direct_inputs.values()), RawText(text=""))
            if isinstance(input_value, FinalAnswer):
                return input_value
            if isinstance(input_value, NotFound):
                return FinalAnswer(text=input_value.detail, sources=[])
            text = getattr(input_value, "text", json.dumps(input_value.model_dump(mode="json")))
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
            return RawText(text=json.dumps(expansion.model_dump(mode="json"), sort_keys=True))
        if node.kind == NodeKind.ROUTER:
            return self._invoke_router(run, node, direct_inputs)
        if node.kind in {NodeKind.MEMORY_QUERY, NodeKind.GOAL, NodeKind.OBSERVATION}:
            raise ExecutionError(f"Unsupported node kind in v0: {node.kind.value}")
        raise ExecutionError(f"Unsupported node kind in v0: {node.kind.value}")

    def _invoke_model_node(self, run: Run, node: Node, direct_inputs: dict[str, BaseModel]) -> BaseModel:
        """Execute a model or compress node, optionally with tool use."""

        selected = select_context(
            storage=self.storage,
            runtime_config=self.runtime_config,
            run=run,
            node=node,
            direct_inputs=direct_inputs,
            embedding_store=self.embedding_store,
            provider=self.provider,
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
        model = str(node.execution.get("model") or self.runtime_config.default_model)
        options = {"model": model}

        tools = self._node_tool_definitions(node)
        if tools:
            result = self._invoke_model_with_tool_loop(run, node, prompt, prompt_artifact.id, tools, options)
        else:
            result = self.provider.generate(prompt, options, None)
            self.storage.persist_model_call(
                ModelCall(
                    id=f"modelcall_{uuid4().hex}",
                    run_id=run.id,
                    node_id=node.id,
                    provider=getattr(self.provider, "name", "provider"),
                    model=model,
                    prompt_artifact_id=prompt_artifact.id,
                    status="succeeded",
                    token_json=result.token_usage or {},
                )
            )
        output = self._coerce_model_output(node.output_schema, result.text)

        # Stash the embedding search query on the node state so callers can
        # inspect what was actually embedded (e.g. after LLM rewrite).
        if selected.search_query:
            state = self._state(run.id, node.id)
            state.metadata["search_query"] = selected.search_query
            self.storage.update_run_node_state(state)
        return output

    def _node_tool_definitions(self, node: Node) -> list[dict[str, Any]] | None:
        """Return the tool definitions a model node should advertise, if any."""

        tools_config = node.execution.get("tools")
        if tools_config is None or tools_config == "none":
            return None
        if tools_config == "all":
            names = self.tool_registry.names()
        elif isinstance(tools_config, list):
            names = list(tools_config)
        else:
            return None
        return [self.tool_registry.to_json_schema(name) for name in names]

    def _invoke_model_with_tool_loop(
        self,
        run: Run,
        node: Node,
        prompt: str,
        prompt_artifact_id: str,
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> ModelOutput:
        """Run a model node with a bounded tool-use loop."""

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        max_rounds = self.runtime_config.max_tool_rounds
        last_result: ModelOutput | None = None
        last_results: list[ToolCallResult] = []

        for _round in range(max_rounds):
            result = self.provider.generate(
                prompt,
                options,
                None,
                tools=tools,
                messages=messages,
            )
            last_result = result
            self.storage.persist_model_call(
                ModelCall(
                    id=f"modelcall_{uuid4().hex}",
                    run_id=run.id,
                    node_id=node.id,
                    provider=getattr(self.provider, "name", "provider"),
                    model=str(options.get("model", self.runtime_config.default_model)),
                    prompt_artifact_id=prompt_artifact_id,
                    status="succeeded",
                    token_json=result.token_usage or {},
                )
            )
            if not result.tool_calls:
                break

            requests = [ToolCallRequest(name=tc.name, arguments=tc.arguments) for tc in result.tool_calls]
            requests_artifact = Artifact(
                id=new_id("artifact"),
                run_id=run.id,
                node_id=node.id,
                port="tool_calls",
                content_json=[req.model_dump(mode="json") for req in requests],
                token_count=self.runtime_config.tokenizer.count_tokens(json.dumps(requests, sort_keys=True, default=str)),
                metadata={"round": _round},
            )
            self.storage.persist_artifact(requests_artifact)

            assistant_message: dict[str, Any] = {"role": "assistant", "content": result.text or ""}
            assistant_message["tool_calls"] = [
                {"function": {"name": req.name, "arguments": req.arguments}} for req in requests
            ]
            messages.append(assistant_message)

            results = self._execute_tool_calls(run, node, requests)
            last_results = results
            results_artifact = Artifact(
                id=new_id("artifact"),
                run_id=run.id,
                node_id=node.id,
                port="tool_results",
                content_json=[res.model_dump(mode="json") for res in results],
                token_count=self.runtime_config.tokenizer.count_tokens(json.dumps(results, sort_keys=True, default=str)),
                metadata={"round": _round},
            )
            self.storage.persist_artifact(results_artifact)

            for req, res in zip(requests, results):
                messages.append(
                    {
                        "role": "tool",
                        "name": req.name,
                        "content": json.dumps(res.model_dump(mode="json"), default=str),
                    }
                )

        if last_result is None:
            raise ExecutionError(f"Tool-use loop produced no result for node {node.name}")
        # If the model never produced a final text response, surface the last tool results.
        if last_result.tool_calls and not last_result.text:
            final_text = json.dumps(
                [res.model_dump(mode="json") for res in last_results],
                sort_keys=True,
                default=str,
            )
            return ModelOutput(
                text=final_text,
                raw=last_result.raw,
                token_usage=last_result.token_usage,
                tool_calls=last_result.tool_calls,
            )
        return last_result

    def _execute_tool_calls(
        self, run: Run, node: Node, requests: list[ToolCallRequest]
    ) -> list[ToolCallResult]:
        """Invoke registered tools for a list of model-requested tool calls."""

        results: list[ToolCallResult] = []
        for req in requests:
            if not self.tool_registry.has(req.name):
                results.append(ToolCallResult(name=req.name, error=f"Unknown tool: {req.name}"))
                continue
            tool = self.tool_registry.get(req.name)
            spec = tool.spec()
            model = self._model_for_schema(spec.input_schema)
            try:
                input_value = model.model_validate(req.arguments)
            except Exception as exc:
                results.append(ToolCallResult(name=req.name, error=f"Invalid arguments: {exc}"))
                continue
            start = perf_counter()
            try:
                output = tool.invoke(input_value)
            except Exception as exc:
                results.append(ToolCallResult(name=req.name, error=f"Tool error: {exc}"))
                continue
            elapsed = int((perf_counter() - start) * 1000)
            output_data: Any
            if isinstance(output, BaseModel):
                output_data = output.model_dump(mode="json")
            else:
                output_data = str(output)
            self.storage.persist_tool_call(
                ToolCall(
                    id=f"toolcall_{uuid4().hex}",
                    run_id=run.id,
                    node_id=node.id,
                    tool_name=req.name,
                    input_json=input_value.model_dump(mode="json"),
                    status="succeeded",
                    latency_ms=elapsed,
                )
            )
            results.append(ToolCallResult(name=req.name, content_json=output_data))
        return results

    def _invoke_router(self, run: Run, node: Node, direct_inputs: dict[str, BaseModel]) -> RoutingDecision:
        mode = node.execution.get("mode", "model")
        if mode == "deterministic":
            input_value = next(iter(direct_inputs.values()), None)
            if isinstance(input_value, NotFound):
                branch = str(node.execution.get("not_found_branch", "missing"))
            else:
                branch = str(node.execution.get("default_branch", "found"))
            return RoutingDecision(selected_branch=branch)
        # model-assisted routing
        branches: list[str] = list(node.execution.get("branches") or [])
        instruction = (
            node.metadata.get("instruction")
            or node.execution.get("prompt_template")
            or f"Classify the input and select one branch: {branches}"
        )
        input_text = ""
        for v in direct_inputs.values():
            candidate = getattr(v, "text", None)
            if candidate:
                input_text = candidate
                break
        if not input_text:
            input_text = json.dumps(
                {name: value.model_dump(mode="json") for name, value in sorted(direct_inputs.items())},
                sort_keys=True,
            )
        branches_str = ", ".join(f'"{b}"' for b in branches)
        prompt = (
            f"{instruction}\n\n"
            f"Input: {input_text}\n\n"
            f"Available branches: [{branches_str}]\n\n"
            f"Respond with ONLY valid JSON in exactly this format (no prose, no markdown):\n"
            f'{{"selected_branch": "<branch_name>"}}'
        )
        result = self.provider.generate(
            prompt,
            {"model": node.execution.get("model") or self.runtime_config.default_model},
            None,
        )
        decision = self._coerce_model_output("RoutingDecision", result.text)
        self.storage.persist_model_call(
            ModelCall(
                id=f"modelcall_{uuid4().hex}",
                run_id=run.id,
                node_id=node.id,
                provider=getattr(self.provider, "name", "provider"),
                model=str(node.execution.get("model") or self.runtime_config.default_model),
                prompt_artifact_id="",
                status="succeeded",
                token_json=result.token_usage or {},
            )
        )
        return decision  # type: ignore[return-value]

    def _gather_inputs(self, run: Run, node: Node) -> tuple[dict[str, BaseModel], list[str]]:
        graph = self.storage.load_task_graph(run.task_graph_id)
        edges = [edge for edge in self.storage.list_task_edges(graph.id) if edge.to_node_id == node.id]
        direct: dict[str, BaseModel] = {}
        input_ids: list[str] = []
        states = {state.node_id: state for state in self.storage.list_run_node_states(run.id)}
        nodes_by_id = {n.id: n for n in graph.nodes}
        for edge in edges:
            from_node = nodes_by_id.get(edge.from_node_id)
            # Router edges are control-only; the RoutingDecision artifact is not
            # data input for the downstream node — skip it.
            if from_node and from_node.kind == NodeKind.ROUTER:
                continue
            state = states.get(edge.from_node_id)
            if not state or not state.output_artifact_ids:
                # A skipped upstream node means the branch wasn't taken — not an error.
                if edge.required and (not state or state.status != NodeStatus.SKIPPED):
                    raise ExecutionError(f"Missing required input {edge.to_port}")
                continue
            artifacts = [self.storage.load_artifact(artifact_id) for artifact_id in state.output_artifact_ids]
            artifact = next((item for item in artifacts if item.port == edge.from_port), artifacts[0])
            input_ids.append(
                sha256(
                    json.dumps(artifact.content_json, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
            )
            value = self._artifact_to_model(from_node.output_schema if from_node else None, artifact.content_json)
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
        # Strip markdown code fences if present
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return model.model_validate_json(stripped)
        except (json.JSONDecodeError, Exception):
            pass
        if model is Summary:
            return Summary(text=text)
        if model is FinalAnswer:
            return FinalAnswer(text=text, sources=[])
        if model is RawText:
            return RawText(text=text)
        if model is RoutingDecision:
            # Model returned a plain word instead of JSON — treat it as the branch name
            return RoutingDecision(selected_branch=stripped.strip('"\' \n\t'))
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
            grounding_mode = node.metadata.get("grounding_mode", "permissive")
            focus_clause = (
                "Focus on the current task above. "
                "Do NOT re-answer earlier questions that may appear in the context."
            )
            if grounding_mode == "strict":
                instruction = (
                    f"{instruction}\n\n"
                    f"--- Context ---\n{prior}\n---\n\n"
                    f"Answer using ONLY the context above. "
                    f"Base your answer on the information in the context, including what it implies or contradicts. "
                    f"If the context contains no relevant information at all, say that the provided context does not contain the answer. "
                    f"Do not use outside knowledge. "
                    f"{focus_clause}"
                )
            else:
                instruction = (
                    f"{instruction}\n\n"
                    f"--- Prior conversation ---\n{prior}\n---\n\n"
                    f"Use the prior conversation above as the primary source. "
                    f"If the answer is in the context above, answer from it. "
                    f"If the context does not contain the answer, you may use your own knowledge. "
                    f"{focus_clause}"
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
