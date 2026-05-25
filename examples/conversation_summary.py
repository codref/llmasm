"""Run the in-memory conversation summary example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmasm.api import LLMASM
from llmasm.config import RuntimeConfig
from llmasm.graph.registry import default_schema_registry
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.registry import ToolRegistry
from tests.unit.fakes import ConversationRetrieveTool, FakeProvider, conversation_summary_proposal


def main() -> None:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    tools.register(ConversationRetrieveTool())
    app = LLMASM(
        storage=InMemoryStorage(),
        provider=FakeProvider([conversation_summary_proposal()], model_text="A concise summary."),
        tool_registry=tools,
        runtime_config=RuntimeConfig(default_model="fake-model"),
        schema_registry=schemas,
    )
    workspace_id = app.create_workspace("example")
    answer = app.ask(workspace_id, "retrieve the conversation xyz and give me a summary of the content")
    print(answer.model_dump())


if __name__ == "__main__":
    main()
