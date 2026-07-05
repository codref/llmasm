## 1. Schemas and Tool Registry

- [x] 1.1 Add `ToolCallRequest`, `ToolCallResult`, and `ToolDefinition` schemas to `llmasm/schemas.py` and register them in the default schema registry.
- [x] 1.2 Add `ToolRegistry.to_json_schema(name)` helper that converts a tool's input Pydantic model to JSON Schema.
- [x] 1.3 Add a built-in `tools.list` introspection tool and ensure it is auto-registered.
- [x] 1.4 Add a decorator/helper in `llmasm/tools/base.py` to build a `Tool` from a Pydantic-typed function.

## 2. Provider Protocol and Ollama Implementation

- [x] 2.1 Update `LLMProvider.generate` protocol in `llmasm/providers/base.py` to accept optional `tools` and return `tool_calls` in `ModelOutput`.
- [x] 2.2 Update `OllamaProvider.generate` to route to `/api/chat` when tools are provided and parse `tool_calls` from the response.
- [x] 2.3 Update `FakeProvider` in `tests/unit/fakes.py` to support scripted tool-call responses.

## 3. Runtime Tool-Use Loop

- [x] 3.1 Add `max_tool_rounds` to `RuntimeConfig` in `llmasm/config.py`.
- [x] 3.2 Implement tool selection logic in `Executor._invoke` for `MODEL`/`COMPRESS` nodes based on `node.execution.get("tools")`.
- [x] 3.3 Implement the request/result loop: collect tool calls, invoke tools, feed results back, cap rounds.
- [x] 3.4 Persist `tool_calls` and `tool_results` artifacts on the model node for each round.
- [x] 3.5 Handle unknown tools and tool-call parse errors gracefully by feeding an error result back to the model.

## 4. Examples

- [x] 4.1 Create `examples/custom_tool.py` demonstrating a custom Pydantic-typed tool, registration, and invocation through the planner path.
- [x] 4.2 Add a `/tools` command to `examples/chat.py` that prints the names and descriptions of registered tools.
- [x] 4.3 Update `examples/chat.py` help text to mention `/tools`.

## 5. Integration and Tests

- [x] 5.1 Add unit tests for `ToolRegistry.to_json_schema` and `tools.list`.
- [x] 5.2 Add unit tests for `OllamaProvider` tool-call request construction and response parsing (using `respx` or mocked httpx).
- [x] 5.3 Add unit tests for the executor tool-use loop covering single call, parallel calls, unknown tool, and max-round exhaustion.
- [x] 5.4 Run `make test`, `make lint`, and `make typecheck`; fix any failures.
  - `make test`: 117 passed, 20 skipped (Postgres tests skipped without LLMASM_TEST_DB).
  - `make lint`: passed.
  - `make typecheck`: 68 pre-existing errors in untouched files (storage, scheduler, expansion, context, validation). No new errors in changed files; fixed one compiler.py error surfaced by the run.
