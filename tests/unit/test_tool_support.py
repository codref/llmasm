"""Tests for tool-use support."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from llmasm.api import LLMASM  # noqa: F401  # import first to initialize module graph
from llmasm.runtime.executor import Executor

from llmasm.config import RuntimeConfig
from llmasm.errors import ValidationError
from llmasm.graph.models import Node, NodeKind, Run, TaskEdge, TaskGraph, WorkspaceGraph
from llmasm.graph.registry import SchemaRegistry, default_schema_registry
from llmasm.graph.transforms import default_transform_registry
from llmasm.ids import new_id
from llmasm.providers.base import ModelOutput, ToolCallOutput
from llmasm.providers.ollama import OllamaProvider
from llmasm.schemas import FinalAnswer, RawText, ToolCallRequest, ToolCallResult
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.base import make_tool
from llmasm.tools.calculator import CalculatorTool
from llmasm.tools.registry import ToolRegistry
from tests.unit.fakes import FakeProvider


class AddInput(BaseModel):
    """Input for the add tool."""

    a: int
    b: int


def add_tool(input: BaseModel) -> BaseModel:
    """Add two numbers."""

    return RawText(text=str(getattr(input, "a", 0) + getattr(input, "b", 0)))


def test_tool_registry_to_json_schema() -> None:
    schemas = default_schema_registry()
    schemas.register("AddInput", AddInput)
    tools = ToolRegistry(schemas)
    tools.register(make_tool(add_tool, AddInput, output_schema="RawText", name="math.add"))

    schema = tools.to_json_schema("math.add")

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "math.add"
    assert "description" in schema["function"]
    assert schema["function"]["parameters"]["type"] == "object"
    assert "a" in schema["function"]["parameters"]["properties"]
    assert "b" in schema["function"]["parameters"]["properties"]


def test_tool_registry_to_json_schema_unknown_tool() -> None:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)

    with pytest.raises(ValidationError):
        tools.to_json_schema("missing.tool")


def test_tools_list_auto_registered() -> None:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)

    assert tools.has("tools.list")
    result = tools.get("tools.list").invoke(RawText(text="ignored"))

    assert isinstance(result.value, list)
    assert any(item["name"] == "tools.list" for item in result.value)


def test_tools_list_includes_registered_tools() -> None:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    tools.register(CalculatorTool())

    result = tools.get("tools.list").invoke(RawText(text="ignored"))

    names = {item["name"] for item in result.value}
    assert "calculator.eval" in names
    assert "tools.list" in names


def _build_tool_node_graph(
    workspace_id: str,
    storage: InMemoryStorage,
    *,
    tool_outputs: list[ModelOutput] | None = None,
    tools_config: Any = "all",
) -> tuple[Run, TaskGraph, FakeProvider, ToolRegistry, SchemaRegistry]:
    """Build intent -> model (with tools) -> final graph and return run + graph + provider + tools + schemas."""

    schemas = default_schema_registry()
    schemas.register("AddInput", AddInput)
    tools = ToolRegistry(schemas)
    tools.register(make_tool(add_tool, AddInput, output_schema="RawText", name="math.add"))

    tg_id = new_id("taskgraph")
    n_intent = Node(
        id=new_id("node"),
        workspace_graph_id=workspace_id,
        task_graph_id=tg_id,
        kind=NodeKind.INTENT,
        name="intent",
        output_schema="RawText",
        metadata={"output": {"text": "add 2 and 3"}},
    )
    n_model = Node(
        id=new_id("node"),
        workspace_graph_id=workspace_id,
        task_graph_id=tg_id,
        kind=NodeKind.MODEL,
        name="answer",
        input_schema="RawText",
        output_schema="RawText",
        execution={"model": "fake-model", "tools": tools_config},
        metadata={"instruction": "Use the add tool if needed."},
    )
    n_final = Node(
        id=new_id("node"),
        workspace_graph_id=workspace_id,
        task_graph_id=tg_id,
        kind=NodeKind.FINAL,
        name="final",
        input_schema="RawText",
        output_schema="FinalAnswer",
    )
    edges = [
        TaskEdge(
            id=new_id("edge"),
            workspace_graph_id=workspace_id,
            task_graph_id=tg_id,
            from_node_id=n_intent.id,
            from_port="output",
            to_node_id=n_model.id,
            to_port="input",
        ),
        TaskEdge(
            id=new_id("edge"),
            workspace_graph_id=workspace_id,
            task_graph_id=tg_id,
            from_node_id=n_model.id,
            from_port="output",
            to_node_id=n_final.id,
            to_port="input",
        ),
    ]
    graph = TaskGraph(
        id=tg_id,
        workspace_graph_id=workspace_id,
        nodes=[n_intent, n_model, n_final],
        task_edges=edges,
    )
    storage.persist_task_graph(graph)
    run = Run(id=new_id("run"), workspace_graph_id=workspace_id, task_graph_id=tg_id)
    storage.create_run(run)

    provider = FakeProvider(tool_outputs=tool_outputs or [])
    return run, graph, provider, tools, schemas


def test_executor_single_tool_call() -> None:
    from llmasm.graph.models import NodeStatus, RunStatus

    storage = InMemoryStorage()
    workspace_id = new_id("workspace")
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))

    tool_outputs = [
        ModelOutput(
            text="",
            tool_calls=[ToolCallOutput(name="math.add", arguments={"a": 2, "b": 3})],
            token_usage={"input_tokens": 10, "output_tokens": 5},
        ),
        ModelOutput(text="5", token_usage={"input_tokens": 20, "output_tokens": 1}),
    ]
    run, graph, provider, tools, schemas = _build_tool_node_graph(
        workspace_id, storage, tool_outputs=tool_outputs, tools_config="all"
    )

    executor = Executor(
        storage=storage,
        tool_registry=tools,
        provider=provider,
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(default_model="fake-model"),
    )
    completed_run = executor.execute(run.id)

    assert completed_run.status == RunStatus.SUCCEEDED
    states = {s.node_id: s for s in storage.list_run_node_states(run.id)}
    assert states[graph.nodes[1].id].status == NodeStatus.SUCCEEDED

    tool_artifacts = [a for a in storage.list_artifacts(run.id) if a.port == "tool_calls"]
    result_artifacts = [a for a in storage.list_artifacts(run.id) if a.port == "tool_results"]
    assert len(tool_artifacts) == 1
    assert len(result_artifacts) == 1

    requests = [ToolCallRequest.model_validate(item) for item in tool_artifacts[0].content_json]
    assert requests[0].name == "math.add"
    assert requests[0].arguments == {"a": 2, "b": 3}

    results = [ToolCallResult.model_validate(item) for item in result_artifacts[0].content_json]
    assert results[0].name == "math.add"
    assert results[0].content_json == {"text": "5"}


def test_executor_parallel_tool_calls() -> None:
    from llmasm.graph.models import RunStatus

    storage = InMemoryStorage()
    workspace_id = new_id("workspace")
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))

    tool_outputs = [
        ModelOutput(
            text="",
            tool_calls=[
                ToolCallOutput(name="math.add", arguments={"a": 1, "b": 2}),
                ToolCallOutput(name="math.add", arguments={"a": 3, "b": 4}),
            ],
            token_usage={"input_tokens": 10, "output_tokens": 10},
        ),
        ModelOutput(text="3, 7", token_usage={"input_tokens": 30, "output_tokens": 3}),
    ]
    run, graph, provider, tools, schemas = _build_tool_node_graph(
        workspace_id, storage, tool_outputs=tool_outputs, tools_config=["math.add"]
    )

    executor = Executor(
        storage=storage,
        tool_registry=tools,
        provider=provider,
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(default_model="fake-model"),
    )
    completed_run = executor.execute(run.id)

    assert completed_run.status == RunStatus.SUCCEEDED
    tool_artifacts = [a for a in storage.list_artifacts(run.id) if a.port == "tool_calls"]
    requests = [ToolCallRequest.model_validate(item) for item in tool_artifacts[0].content_json]
    assert len(requests) == 2


def test_executor_unknown_tool_call() -> None:
    from llmasm.graph.models import RunStatus

    storage = InMemoryStorage()
    workspace_id = new_id("workspace")
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))

    tool_outputs = [
        ModelOutput(
            text="",
            tool_calls=[ToolCallOutput(name="missing.tool", arguments={})],
            token_usage={"input_tokens": 10, "output_tokens": 5},
        ),
        ModelOutput(text="done", token_usage={"input_tokens": 20, "output_tokens": 1}),
    ]
    run, graph, provider, tools, schemas = _build_tool_node_graph(
        workspace_id, storage, tool_outputs=tool_outputs, tools_config="all"
    )

    executor = Executor(
        storage=storage,
        tool_registry=tools,
        provider=provider,
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(default_model="fake-model"),
    )
    completed_run = executor.execute(run.id)

    assert completed_run.status == RunStatus.SUCCEEDED
    result_artifacts = [a for a in storage.list_artifacts(run.id) if a.port == "tool_results"]
    results = [ToolCallResult.model_validate(item) for item in result_artifacts[0].content_json]
    assert results[0].error == "Unknown tool: missing.tool"


def test_executor_max_tool_rounds_exhausted() -> None:
    from llmasm.graph.models import RunStatus

    storage = InMemoryStorage()
    workspace_id = new_id("workspace")
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))

    # The model always requests a tool, so the loop should cap at max_tool_rounds.
    tool_outputs = [
        ModelOutput(
            text="",
            tool_calls=[ToolCallOutput(name="math.add", arguments={"a": 1, "b": 1})],
            token_usage={"input_tokens": 10, "output_tokens": 5},
        )
    ] * 10
    run, graph, provider, tools, schemas = _build_tool_node_graph(
        workspace_id, storage, tool_outputs=tool_outputs, tools_config="all"
    )

    executor = Executor(
        storage=storage,
        tool_registry=tools,
        provider=provider,
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(default_model="fake-model", max_tool_rounds=2),
    )
    completed_run = executor.execute(run.id)

    assert completed_run.status == RunStatus.SUCCEEDED
    tool_artifacts = [a for a in storage.list_artifacts(run.id) if a.port == "tool_calls"]
    assert len(tool_artifacts) == 2


def test_executor_tool_use_disabled_by_default() -> None:
    """A model node without a tools config should not enter the tool loop."""

    from llmasm.graph.models import NodeStatus, RunStatus

    storage = InMemoryStorage()
    workspace_id = new_id("workspace")
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))

    run, graph, provider, tools, schemas = _build_tool_node_graph(
        workspace_id, storage, tool_outputs=[], tools_config=None
    )
    provider.model_text = "plain answer"

    executor = Executor(
        storage=storage,
        tool_registry=tools,
        provider=provider,
        schema_registry=schemas,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(default_model="fake-model"),
    )
    completed_run = executor.execute(run.id)

    assert completed_run.status == RunStatus.SUCCEEDED
    states = {s.node_id: s for s in storage.list_run_node_states(run.id)}
    assert states[graph.nodes[1].id].status == NodeStatus.SUCCEEDED
    tool_artifacts = [a for a in storage.list_artifacts(run.id) if a.port == "tool_calls"]
    assert not tool_artifacts
    final_answer = next(a for a in storage.list_artifacts(run.id) if a.node_id == graph.nodes[2].id)
    assert FinalAnswer.model_validate(final_answer.content_json).text == "plain answer"


class _MockResponse:
    def __init__(self, json_data: dict[str, Any]) -> None:
        self._json = json_data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._json


def test_ollama_provider_generate_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def mock_post(url: str, **kwargs: Any) -> _MockResponse:
        calls.append({"url": url, "json": kwargs.get("json")})
        return _MockResponse({"response": "hello", "prompt_eval_count": 3, "eval_count": 1})

    monkeypatch.setattr(httpx, "post", mock_post)
    provider = OllamaProvider()

    result = provider.generate("say hi")

    assert result.text == "hello"
    assert result.tool_calls == []
    assert calls[0]["url"] == "http://localhost:11434/api/generate"


def test_ollama_provider_chat_with_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def mock_post(url: str, **kwargs: Any) -> _MockResponse:
        calls.append({"url": url, "json": kwargs.get("json")})
        return _MockResponse(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "math.add", "arguments": {"a": 1, "b": 2}}}
                    ],
                },
                "prompt_eval_count": 5,
                "eval_count": 10,
            }
        )

    monkeypatch.setattr(httpx, "post", mock_post)
    provider = OllamaProvider()
    tools = [
        {
            "type": "function",
            "function": {"name": "math.add", "description": "Add numbers", "parameters": {}},
        }
    ]

    result = provider.generate("add 1 and 2", tools=tools)

    assert result.text == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "math.add"
    assert result.tool_calls[0].arguments == {"a": 1, "b": 2}
    assert calls[0]["url"] == "http://localhost:11434/api/chat"
    sent = calls[0]["json"]
    assert sent["tools"] == tools
    assert sent["messages"][0]["role"] == "user"


def test_ollama_provider_chat_follow_up_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def mock_post(url: str, **kwargs: Any) -> _MockResponse:
        calls.append({"url": url, "json": kwargs.get("json")})
        return _MockResponse(
            {
                "message": {"role": "assistant", "content": "The answer is 3"},
                "prompt_eval_count": 4,
                "eval_count": 3,
            }
        )

    monkeypatch.setattr(httpx, "post", mock_post)
    provider = OllamaProvider()
    messages = [
        {"role": "user", "content": "add"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "math.add"}}]},
        {"role": "tool", "content": "3"},
    ]

    result = provider.generate("add", messages=messages)

    assert result.text == "The answer is 3"
    assert calls[0]["json"]["messages"] == messages
