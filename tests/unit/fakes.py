"""Unit-test fakes."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from llmasm.providers.base import EmbeddingOutput, ModelInfo, ModelOutput
from llmasm.schemas import ConversationRecord
from llmasm.tools.base import ToolSpec


class FakeProvider:
    """Deterministic provider used by tests."""

    name = "fake"

    def __init__(
        self,
        planner_outputs: list[str] | None = None,
        model_text: str = "summary text",
        tool_outputs: list[ModelOutput] | None = None,
    ) -> None:
        self.planner_outputs = list(planner_outputs or [])
        self.model_text = model_text
        self.tool_outputs = list(tool_outputs or [])
        self.generate_prompts: list[str] = []
        self.embed_calls = 0
        self.generate_calls: list[dict[str, Any]] = []

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="fake-model", context_window=4096), ModelInfo(name="llama3.1:8b", context_window=8192)]

    def generate(
        self,
        prompt: str,
        options: dict[str, Any] | None = None,
        format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        self.generate_prompts.append(prompt)
        self.generate_calls.append({"tools": tools, "messages": messages})
        if format_schema is not None and self.planner_outputs:
            return ModelOutput(text=self.planner_outputs.pop(0), token_usage={"input_tokens": 1, "output_tokens": 1})
        if tools is not None and self.tool_outputs:
            output = self.tool_outputs.pop(0)
            return ModelOutput(
                text=output.text,
                raw=output.raw,
                token_usage=output.token_usage or {"input_tokens": 5, "output_tokens": 2},
                tool_calls=output.tool_calls,
            )
        text = self.model_text(prompt) if callable(self.model_text) else self.model_text
        return ModelOutput(text=text, token_usage={"input_tokens": 5, "output_tokens": 2})

    def embed(self, texts: list[str], options: dict[str, Any] | None = None) -> list[EmbeddingOutput]:
        self.embed_calls += len(texts)
        return [EmbeddingOutput(vector=[float(len(text)), 1.0]) for text in texts]


class ConversationRetrieveTool:
    """Fake conversation retriever."""

    def __init__(self) -> None:
        self.calls = 0

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="conversation_store.retrieve",
            description="Retrieve a conversation by id",
            input_schema="RawText",
            output_schema="ConversationRecord",
        )

    def invoke(self, input: BaseModel) -> BaseModel:
        self.calls += 1
        text = getattr(input, "text", "")
        return ConversationRecord(
            id=text or "xyz",
            text="Project scope discussed graph persistence and local Ollama execution.",
            metadata={"conversation_id": text or "xyz"},
        )


def conversation_summary_proposal() -> str:
    """Return a valid summary planner proposal."""

    return json.dumps(
        {
            "intent": "summarize conversation",
            "goal_action": "new",
            "goal_update_text": "Summarize conversation xyz",
            "nodes": [
                {
                    "name": "intent",
                    "kind": "intent",
                    "output_schema": "RawText",
                    "execution": {"options": {}, "allow_cache": False},
                    "metadata": {"output": {"text": "xyz"}},
                },
                {
                    "name": "retrieve",
                    "kind": "tool",
                    "input_schema": "RawText",
                    "output_schema": "ConversationRecord",
                    "execution": {"tool": "conversation_store.retrieve", "allow_cache": True},
                },
                {
                    "name": "summarize",
                    "kind": "model",
                    "input_schema": "ConversationText",
                    "output_schema": "Summary",
                    "execution": {"provider": "fake", "model": "fake-model", "allow_cache": False},
                },
                {
                    "name": "final",
                    "kind": "final",
                    "input_schema": "Summary",
                    "output_schema": "FinalAnswer",
                },
            ],
            "edges": [
                {"from_node": "intent", "from_port": "output", "to_node": "retrieve", "to_port": "input"},
                {
                    "from_node": "retrieve",
                    "from_port": "output",
                    "to_node": "summarize",
                    "to_port": "input",
                    "transform": "extract_text",
                },
                {"from_node": "summarize", "from_port": "output", "to_node": "final", "to_port": "input"},
            ],
        }
    )
