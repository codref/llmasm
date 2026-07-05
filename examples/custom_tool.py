"""Example: author a custom tool and invoke it through a model node.

This script shows how to:
  1. Define a Pydantic input model for a tool.
  2. Build a Tool from a typed function.
  3. Register the tool with LLMASM.
  4. Run a task graph whose model node advertises the tool to the LLM.

It uses the deterministic FakeProvider so it runs without a local Ollama
server. In production you would swap FakeProvider for OllamaProvider and
let the planner emit the model node, or construct the task graph yourself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmasm.api import LLMASM
from llmasm.config import RuntimeConfig
from llmasm.graph.models import Node, NodeKind, TaskEdge, TaskGraph
from llmasm.graph.registry import default_schema_registry
from llmasm.ids import new_id
from llmasm.providers.base import ModelOutput, ToolCallOutput
from llmasm.schemas import FinalAnswer, RawText
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.base import make_tool
from llmasm.tools.registry import ToolRegistry
from tests.unit.fakes import FakeProvider


class ReverseInput(BaseModel):
    """Input for the reverse_text tool."""

    text: str


def reverse_text(input: BaseModel) -> BaseModel:
    """Reverse the input text."""

    text = getattr(input, "text", "")
    return RawText(text=text[::-1])


def main() -> None:
    schema_registry = default_schema_registry()
    schema_registry.register("ReverseInput", ReverseInput)
    tools = ToolRegistry(schema_registry)
    tools.register(make_tool(reverse_text, ReverseInput, output_schema="RawText", name="reverse.text"))

    # FakeProvider is scripted to first request the reverse.text tool, then
    # answer with the reversed result.
    provider = FakeProvider(
        tool_outputs=[
            ModelOutput(
                text="",
                token_usage={"input_tokens": 10, "output_tokens": 15},
                tool_calls=[ToolCallOutput(name="reverse.text", arguments={"text": "hello"})],
            ),
            ModelOutput(
                text="The reversed text is: olleh",
                token_usage={"input_tokens": 20, "output_tokens": 6},
            ),
        ]
    )

    storage = InMemoryStorage()
    app = LLMASM(
        storage=storage,
        provider=provider,
        tool_registry=tools,
        runtime_config=RuntimeConfig(default_model="fake-model"),
        schema_registry=schema_registry,
    )

    workspace_id = app.create_workspace("custom-tool-demo")
    tg_id = new_id("taskgraph")
    n_intent = Node(
        id=new_id("node"),
        workspace_graph_id=workspace_id,
        task_graph_id=tg_id,
        kind=NodeKind.INTENT,
        name="intent",
        output_schema="RawText",
        metadata={"output": {"text": "Reverse the word hello"}},
    )
    n_model = Node(
        id=new_id("node"),
        workspace_graph_id=workspace_id,
        task_graph_id=tg_id,
        kind=NodeKind.MODEL,
        name="answer",
        input_schema="RawText",
        output_schema="RawText",
        execution={"model": "fake-model", "tools": "all"},
        metadata={"instruction": "Reverse the input word using the available tool."},
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

    run_id = app.run(tg_id)
    analysis = app.query_run(run_id)
    final_node = next(n for n in graph.nodes if n.kind == NodeKind.FINAL)
    final_artifacts = [a for a in analysis.artifacts if a.node_id == final_node.id]
    if final_artifacts:
        answer = FinalAnswer.model_validate(final_artifacts[-1].content_json)
        print(json.dumps(answer.model_dump(mode="json"), indent=2))
    else:
        print("No final answer produced")


if __name__ == "__main__":
    main()
