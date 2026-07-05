## Context

LLMASM today treats tools as static task-graph nodes. The planner emits a `TOOL` node, the executor invokes the registered tool, and the result flows downstream. This works for simple, planner-determined workflows but cannot handle cases where the model itself must decide to call a tool mid-generation, observe the result, and continue reasoning. Ollama supports this via the `/api/chat` endpoint with a `tools` array and `tool_calls` in the response. We will expose that capability through the provider protocol and add a request/result loop in the executor's model-node path.

## Goals / Non-Goals

**Goals:**
- Allow any registered tool to be advertised to a model node as a JSON Schema function definition.
- Support single and parallel tool calls within one model generation.
- Return tool results back to the model as follow-up messages so it can produce a final answer.
- Keep the existing planner-driven `TOOL` node behavior unchanged.
- Maintain backward compatibility for provider calls that do not use tools.

**Non-Goals:**
- Streaming responses from the model.
- Multi-turn conversation state in the chat endpoint (we send a constructed message list, not a persisted conversation).
- Tool-call authentication or secrets management beyond what tools already handle.
- Changing the planner to prefer dynamic tool use over static tool nodes.
- Tool use in the fast-path `chat()` method; the fast path remains a simple `intent -> model -> final` pipeline.

## Decisions

1. **Provider signature change**
   - Add an optional `tools: list[dict[str, Any]] | None = None` parameter to `LLMProvider.generate`.
   - Rationale: Keeps the protocol generic while letting providers opt in. `OllamaProvider` will use `/api/chat` whenever `tools` is non-empty.
   - Alternative considered: A separate `chat` method on the provider. Rejected because it duplicates message-handling logic and forces callers to branch.

2. **Message format inside the provider**
   - Construct a single `user` message containing the prompt and append `tool` messages for results. Tool calls are sent back as `assistant` messages with `tool_calls`.
   - Rationale: Minimal mapping to Ollama's expected chat format; avoids introducing a full conversation history model.

3. **Loop location**
   - Implement the tool-use loop in `Executor._invoke` for `MODEL`/`COMPRESS` nodes, capping iterations via `RuntimeConfig.max_tool_rounds` (default 5).
   - Rationale: The executor already owns node lifecycle, artifact persistence, and tool invocation. Centralizing here avoids scattering state across the provider.

4. **Tool definition generation**
   - Add `ToolRegistry.to_json_schema(name)` and a helper on `ToolSpec` that derives the JSON Schema from the Pydantic model referenced by `input_schema`.
   - Rationale: Prevents schema drift between tool inputs and what the model sees.

5. **Result artifact shape**
   - Store each `ToolCallRequest` and `ToolCallResult` as artifacts with ports `tool_calls` and `tool_results` on the model node.
   - Rationale: Preserves observability and integrates with the existing artifact/audit model (`ToolCall` records are still persisted).

6. **Node-level opt-in**
   - Tool availability is controlled by `node.execution.get("tools", "all")` which can be `"all"`, `"none"`, or a list of tool names.
   - Rationale: Lets the planner decide which tools a specific model node may use without changing global configuration.

7. **Examples for manual exploration**
   - Add `examples/custom_tool.py` showing a minimal custom tool from a Pydantic model and function.
   - Add a `/tools` REPL command to `examples/chat.py` that prints registered tool names and descriptions.
   - Keep the fast-path `chat()` mode unchanged so manual testing remains predictable.
   - Rationale: The user wants to exercise the library manually through `chat.py`; tooling introspection and a self-contained custom-tool example make that easier without complicating the fast path.

## Risks / Trade-offs

- **[Risk]** Switching from `/api/generate` to `/api/chat` when tools are present may produce subtly different formatting behavior for the same model.  
  **Mitigation:** Keep the no-tools path on `/api/generate`; only use `/api/chat` when tools are supplied.

- **[Risk]** Models may infinite-loop calling tools.  
  **Mitigation:** Hard cap on tool rounds; after the cap, the last tool results are summarized to a `RawText` output and the node succeeds.

- **[Risk]** Tool-call JSON parsing errors from the LLM.  
  **Mitigation:** Wrap parsing in try/except; return an error `ToolCallResult` so the model can retry or stop.

- **[Risk]** `LLMProvider` is a Protocol; adding a parameter is a breaking change for any out-of-tree provider.  
  **Mitigation:** Document as breaking; update `FakeProvider` and tests in the same change.

## Migration Plan

No data migration is required. Existing task graphs without tool-enabled model nodes continue to use `/api/generate` and existing `TOOL` nodes. Custom providers outside the repo must add the `tools` parameter to their `generate` method.

## Open Questions

- Should tool results also be embedded into the workspace memory, or kept only as artifacts?
