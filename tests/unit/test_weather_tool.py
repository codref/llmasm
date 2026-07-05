"""Tests for the Weather example tool."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from llmasm.api import LLMASM  # noqa: F401  # import first to initialize module graph
from llmasm.config import RuntimeConfig
from llmasm.graph.models import Node, NodeKind, Run, TaskEdge, TaskGraph, WorkspaceGraph
from llmasm.graph.registry import default_schema_registry
from llmasm.graph.transforms import default_transform_registry
from llmasm.ids import new_id
from llmasm.runtime.executor import Executor
from llmasm.schemas import FinalAnswer, RawText, WeatherObservation
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.registry import ToolRegistry
from llmasm.tools.weather import WeatherTool
from tests.unit.fakes import FakeProvider


class _MockResponse:
    def __init__(self, json_data: dict[str, Any] | None = None, raise_error: Exception | None = None) -> None:
        self._json = json_data or {}
        self._raise = raise_error

    def raise_for_status(self) -> None:
        if self._raise is not None:
            raise self._raise

    def json(self) -> dict[str, Any]:
        return self._json


def _mock_get(responses: list[_MockResponse]) -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []
    index = 0

    def mock_get(url: str, **kwargs: Any) -> _MockResponse:
        nonlocal index
        calls.append({"url": url, "params": kwargs.get("params", {}), "headers": kwargs.get("headers", {})})
        response = responses[index]
        index = min(index + 1, len(responses) - 1)
        return response

    return calls, mock_get


def _geocoding_response() -> dict[str, Any]:
    return {
        "results": [
            {
                "latitude": 52.52437,
                "longitude": 13.41053,
                "name": "Berlin",
            }
        ]
    }


def _forecast_response() -> dict[str, Any]:
    return {
        "current_weather": {
            "temperature": 22.2,
            "windspeed": 20.1,
            "weathercode": 3,
        }
    }


def test_weather_tool_location_found(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, mock_get = _mock_get(
        [
            _MockResponse(_geocoding_response()),
            _MockResponse(_forecast_response()),
        ]
    )
    monkeypatch.setattr(httpx, "get", mock_get)

    result = WeatherTool().invoke(RawText(text="Berlin"))

    assert isinstance(result, WeatherObservation)
    assert "Berlin" in result.condition
    assert "22.2°C" in result.condition
    assert "overcast" in result.condition
    assert result.source_url is not None
    assert "open-meteo.com" in result.source_url


def test_weather_tool_location_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, mock_get = _mock_get(
        [
            _MockResponse({"results": []}),
        ]
    )
    monkeypatch.setattr(httpx, "get", mock_get)

    result = WeatherTool().invoke(RawText(text="NotARealPlace12345"))

    assert isinstance(result, WeatherObservation)
    assert "not found" in result.condition


def test_weather_tool_empty_input() -> None:
    result = WeatherTool().invoke(RawText(text="   "))

    assert isinstance(result, WeatherObservation)
    assert "No location provided" in result.condition


def test_weather_tool_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, mock_get = _mock_get(
        [
            _MockResponse(raise_error=httpx.HTTPError("network down")),
        ]
    )
    monkeypatch.setattr(httpx, "get", mock_get)

    result = WeatherTool().invoke(RawText(text="Berlin"))

    assert isinstance(result, WeatherObservation)
    assert "Weather API error" in result.condition


def test_weather_tool_static_node_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, mock_get = _mock_get(
        [
            _MockResponse(_geocoding_response()),
            _MockResponse(_forecast_response()),
        ]
    )
    monkeypatch.setattr(httpx, "get", mock_get)

    schema_registry = default_schema_registry()
    tools = ToolRegistry(schema_registry)
    tools.register(WeatherTool())

    storage = InMemoryStorage()
    workspace_id = new_id("workspace")
    storage.create_workspace_graph(WorkspaceGraph(id=workspace_id, name="test"))

    tg_id = new_id("taskgraph")
    n_intent = Node(
        id=new_id("node"),
        workspace_graph_id=workspace_id,
        task_graph_id=tg_id,
        kind=NodeKind.INTENT,
        name="intent",
        output_schema="RawText",
        metadata={"output": {"text": "Berlin"}},
    )
    n_tool = Node(
        id=new_id("node"),
        workspace_graph_id=workspace_id,
        task_graph_id=tg_id,
        kind=NodeKind.TOOL,
        name="weather",
        input_schema="RawText",
        output_schema="WeatherObservation",
        execution={"tool": "weather.lookup", "allow_cache": False},
    )
    n_model = Node(
        id=new_id("node"),
        workspace_graph_id=workspace_id,
        task_graph_id=tg_id,
        kind=NodeKind.MODEL,
        name="answer",
        input_schema="WeatherObservation",
        output_schema="FinalAnswer",
        execution={"model": "fake-model"},
        metadata={"instruction": "Summarize the weather."},
    )
    n_final = Node(
        id=new_id("node"),
        workspace_graph_id=workspace_id,
        task_graph_id=tg_id,
        kind=NodeKind.FINAL,
        name="final",
        input_schema="FinalAnswer",
        output_schema="FinalAnswer",
    )
    edges = [
        TaskEdge(
            id=new_id("edge"),
            workspace_graph_id=workspace_id,
            task_graph_id=tg_id,
            from_node_id=n_intent.id,
            from_port="output",
            to_node_id=n_tool.id,
            to_port="input",
        ),
        TaskEdge(
            id=new_id("edge"),
            workspace_graph_id=workspace_id,
            task_graph_id=tg_id,
            from_node_id=n_tool.id,
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
        root_prompt_node_id=n_intent.id,
        nodes=[n_intent, n_tool, n_model, n_final],
        task_edges=edges,
    )
    storage.persist_task_graph(graph)

    run = Run(id=new_id("run"), workspace_graph_id=workspace_id, task_graph_id=tg_id)
    storage.create_run(run)

    provider = FakeProvider(model_text="It is overcast in Berlin.")
    executor = Executor(
        storage=storage,
        tool_registry=tools,
        provider=provider,
        schema_registry=schema_registry,
        transform_registry=default_transform_registry(),
        runtime_config=RuntimeConfig(default_model="fake-model"),
    )
    completed_run = executor.execute(run.id)

    assert completed_run.status == "succeeded"
    final_artifacts = [a for a in storage.list_artifacts(run.id) if a.node_id == n_final.id]
    assert final_artifacts
    answer = FinalAnswer.model_validate(final_artifacts[-1].content_json)
    assert "Berlin" in answer.text
