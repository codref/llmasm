## Why

LLMASM already has a `ToolRegistry` and `TOOL` nodes, but the only way for a model to use a tool is for the planner to emit a dedicated tool node in the task graph. This prevents model nodes from making dynamic tool calls during reasoning (function calling), which is essential for multi-step retrieval, calculation, and external lookups within a single turn. Adding native tool-use support lets models request tool invocations and receive results inline, closing the gap between static planner graphs and interactive agent behavior.

## What Changes

- Extend `LLMProvider` and `OllamaProvider` to support tool definitions and tool-call responses via Ollama's chat API.
- Add a tool-use loop inside model-node execution so a model can emit one or more tool calls, the runtime invokes them, and results are fed back as follow-up messages.
- Introduce a `ToolCallRequest` / `ToolCallResult` schema pair and register them in the default schema registry.
- Expose registered tools to the planner/compiler as JSON Schema so the model knows which tools are available.
- Add a built-in `tools.list` introspection tool and convenience helpers for registering tools with typed Pydantic input/output models.
- Keep the fast-path `chat()` method tool-free; instead surface tools through the planner path and provide example scripts under `examples/` for manual exploration.
- Add `examples/custom_tool.py` demonstrating how to author, register, and invoke a custom tool, plus a `/tools` command in `examples/chat.py` to list available tools.
- Add unit tests covering single tool calls, parallel tool calls, missing-tool handling, and cache interaction.

## Capabilities

### New Capabilities
- `tool-use-provider`: Provider-level tool definition and tool-call parsing for Ollama.
- `tool-use-runtime`: Runtime loop that executes tool calls requested by model nodes and returns results.
- `tool-introspection`: Built-in tool for listing available tools and a helper to convert `ToolSpec` to JSON Schema.

### Modified Capabilities
- `model-node`: Model nodes gain optional tool availability; when tools are provided, execution enters a request/result loop instead of returning a single response.

## Impact

- `llmasm/providers/base.py` — `LLMProvider.generate` signature changes to accept optional `tools` parameter. **BREAKING** for any custom provider implementations.
- `llmasm/providers/ollama.py` — Switches model-node generation from `/api/generate` to `/api/chat` when tools are present; adds tool-call parsing.
- `llmasm/runtime/executor.py` — Model-node execution path extended with tool-call loop.
- `llmasm/schemas.py` — New `ToolCallRequest`, `ToolCallResult`, and `ToolDefinition` schemas.
- `llmasm/tools/base.py` and `llmasm/tools/registry.py` — Helpers to emit JSON Schema descriptions and a `tools.list` built-in tool.
- `tests/unit/fakes.py` — `FakeProvider` updated to support tool calls.
