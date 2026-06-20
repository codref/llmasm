# Next Phase Plan: hermes-agent tools + LiteLLM patterns for LLMASM

**Status:** Draft plan — not yet implemented  
**Goal:** Strengthen LLMASM's **tools layer** and **provider layer** with proven patterns from `hermes-agent` (rich, safe tool ecosystem) and `LiteLLM` (unified provider/model abstraction), while preserving LLMASM's graph-first, persistent execution model.

This plan is a follow-up to `hermes-agent-execution-practices-plan.md`, which covers execution-time hardening. This document focuses on tool registration, dispatch, safety, and provider generalization.

---

## Part 1 — Best practices from hermes-agent tools

These come from direct code review of `tools/registry.py`, `toolsets.py`, `model_tools.py`, `agent/tool_executor.py`, `tools/approval.py`, `tools/code_execution_tool.py`, `tools/async_delegation.py`, and related files.

### 1.1 Registration & discovery

| Practice | Why it matters | Maps to LLMASM |
|---|---|---|
| **Self-registering modules + AST-based discovery** | Tool authors cannot forget to wire a new tool; helper modules are excluded. | Add an AST scanner over `llmasm/tools/` or plugin dirs. |
| **`ToolEntry` metadata record** (`__slots__`) | Single source of truth for name, schema, handler, toolset, availability, limits. | Wrap `Tool`/`ToolSpec` in a registry entry. |
| **Toolset as first-class grouping** with includes/aliases | Each actor (CLI, subagent, compiler) declares only the tools it needs. | Add `Toolset` concept and `enabled_toolsets` to `RuntimeConfig`. |
| **Override protection** (`register(..., override=False)`) | Prevents plugins/MCP from silently replacing core tools. | Add `override` flag to `ToolRegistry.register()`. |
| **Registry generation counter** | Safe memoization across dynamic registry mutations. | Add `_generation` to `ToolRegistry`. |

### 1.2 Dispatch & execution

| Practice | Why it matters | Maps to LLMASM |
|---|---|---|
| **Single `registry.dispatch()` returning JSON strings** | One place for tracing, middleware, error normalization. | Consolidate dispatch in `ToolRegistry.dispatch()`. |
| **Async bridging with persistent loops** | Avoids "event loop closed" with cached async clients. | Add a loop bridge if async tools are introduced. |
| **Parallel tool execution with safety gating** | Read-only tools can run concurrently; file-mutating tools must be path-disjoint. | Add parallelism policy to scheduler. |
| **Tool request/execution middleware** | Clean plugin interception without monkey-patching. | Add middleware seams in executor. |
| **Loop-handled tools** (`todo`, `memory`, `delegate`) | Some tools need agent-level state. | Mark tools with `loop_handled` flag. |

### 1.3 Schema & arguments

| Practice | Why it matters | Maps to LLMASM |
|---|---|---|
| **`check_fn` availability gating with TTL cache** | Only expose tools that can actually run. | Add `availability_check` to `ToolSpec`. |
| **Dynamic schema overrides** | Reflect live config without rebuilding registry. | Allow `dynamic_schema` callable. |
| **Schema sanitizer for strict backends** | Local backends reject schemas cloud providers accept. | Add sanitizer pass before sending schemas to providers. |
| **Runtime argument coercion** | Models often emit strings for numbers/booleans. | Coerce args before `tool.invoke()`. |
| **No hardcoded cross-tool references** | Prevents hallucinated calls to disabled tools. | Strip/inject references based on active toolset. |

### 1.4 Safety & sandboxing

| Practice | Why it matters | Maps to LLMASM |
|---|---|---|
| **Layered approval: hardline + pattern + smart** | Defense in depth for dangerous commands. | Add `DangerousCommandGuard` middleware. |
| **Env scrubbing with explicit allowlist** | Child processes do not inherit secrets by default. | Add env-scrubbing helper for code tools. |
| **Sandbox tool allowlist** | Prevent scripts from calling agent-loop tools. | Define fixed allowlist for programmatic tool calls. |
| **Resource limits and output caps** | Bound runaway scripts. | Add timeout/output budgets to tool nodes. |

### 1.5 Observability

| Practice | Why it matters | Maps to LLMASM |
|---|---|---|
| **`pre_tool_call` / `post_tool_call` / `transform_tool_result` hooks** | Extensibility without core changes. | Add hook registry. |
| **Progress callbacks and heartbeats** | Long-running tools do not look idle. | Expose lifecycle callbacks. |
| **Duration measurement** | Latency dashboards and alerts. | Record timestamps in tool artifacts. |
| **Untrusted result wrapping** | Mitigates indirect prompt injection. | Wrap untrusted tool results. |

---

## Part 2 — Best practices from LiteLLM

LiteLLM is not a tool framework; it is a **provider/model unification layer**. The transferable practices are mostly about how to talk to many models/providers reliably.

### 2.1 Unified provider interface

| Practice | Why it matters | Maps to LLMASM |
|---|---|---|
| **Single `completion()` API across 100+ providers** | One call site, many backends. | Generalize `LLMProvider` so Ollama is one implementation of a common interface. |
| **Model string conventions** (`provider/model`) | Clear provider identity. | Adopt `ollama/llama3.2`, `openai/gpt-4o`, etc. in `RuntimeConfig`. |
| **Capability introspection** (`supports_function_calling`, `supports_response_schema`, `get_supported_openai_params`) | Avoid sending unsupported features to a model. | Query provider before enabling tools/JSON schema. |

### 2.2 Exception mapping

| Practice | Why it matters | Maps to LLMASM |
|---|---|---|
| **Map all provider errors to standard error types** | Caller handles one exception taxonomy. | Add `RetryableError`, `ContextWindowError`, `ContentPolicyError`, `RateLimitError`, `BadRequestError`. |
| **Attach `status_code`, `llm_provider`, `provider_specific_fields`** | Richer debugging and fallback decisions. | Include these in LLMASM exceptions. |
| **`_should_retry(status_code)` helper** | Removes guesswork from retry logic. | Use in executor retry/fallback chain. |

### 2.3 Reliability patterns

| Practice | Why it matters | Maps to LLMASM |
|---|---|---|
| **Categorized fallbacks** (`fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`) | Different failures need different recovery. | Add per-error-type fallback chains in `RuntimeConfig`. |
| **Retries + timeouts + cooldowns** | Hides transient failures. | Align with LiteLLM's `num_retries`, `request_timeout`, `allowed_fails`. |
| **Pre-call context-window checks** | Fail fast before burning tokens. | Check token count against model limit before provider call. |
| **Drop unsupported params** (`drop_params=True`) | Prevents provider rejection. | Add provider capability filtering. |

### 2.4 Tool-calling message hygiene

| Practice | Why it matters | Maps to LLMASM |
|---|---|---|
| **`modify_params` message sanitization** | Fixes orphaned tool calls/results and empty content. | Add pre-flight integrity repair. |
| **Client-side JSON schema validation** | Catches malformed structured output. | Validate model output against output schema. |

### 2.5 Guardrails / hooks

| Practice | Why it matters | Maps to LLMASM |
|---|---|---|
| **Guardrails as `pre_call`, `post_call`, `during_call` hooks** | Input/output safety checks. | Generalize hermes middleware into a guardrail layer. |
| **`default_on` guardrails** | Safety policies run without user action. | Support default-enabled guards. |
| **Per-request / per-key guardrail selection** | Flexible policy application. | Allow guardrails to be selected per `Run`. |

---

## Part 3 — Synthesized recommendations for LLMASM

Combine the two sources into a coherent next phase.

### A. Tool layer (hermes-first)

1. **`ToolEntry` wrapper in `ToolRegistry`**  
   Add toolset, availability check, dynamic schema, trusted/untrusted flag, max result size, async flag.

2. **Toolset abstraction**  
   `Toolset` dataclass, `resolve_toolset()` with cycle detection, `enabled_toolsets` in `RuntimeConfig`. The compiler only sees tools from enabled toolsets.

3. **Auto-discovery**  
   AST scan `llmasm/tools/` for `registry.register()` calls; keep `tools/__init__.py` import-free.

4. **Availability gating**  
   Optional `check_fn` on `ToolSpec`, TTL-cached in registry, used when building descriptions for the compiler.

5. **Schema sanitation**  
   Normalize schemas before sending to provider (inject `properties`, collapse nullable unions, strip `$ref` siblings).

6. **Argument coercion**  
   Coerce string args to declared types before invocation.

7. **Error normalization**  
   `ToolRegistry.dispatch()` always returns a JSON-string result; exceptions become `{"error": ...}` after sanitization.

8. **Approval/guardrails middleware**  
   Layered guard: hardline blocklist → dangerous pattern approval → optional smart approval. Pluggable approval backend (CLI callback vs. event queue).

9. **Parallelism policy**  
   Read-only tools may run concurrently; file-mutating tools are path-disjoint or sequential.

10. **Untrusted result wrapping**  
    Wrap results from web/browser/MCP tools before adding to context.

### B. Provider layer (LiteLLM-first)

1. **Generalize `LLMProvider`**  
   Support model strings like `ollama/llama3.2`, `openai/gpt-4o`, etc. Add capability queries: `supports_function_calling()`, `supports_response_schema()`, `supports_tools()`.

2. **Add LiteLLM-backed provider**  
   Implement `LiteLLMProvider` that delegates to `litellm.completion()` / `litellm.embedding()`. This immediately unlocks 100+ providers without custom implementations.

3. **Standardized exception taxonomy**  
   `RetryableError`, `ContextWindowExceededError`, `ContentPolicyViolationError`, `RateLimitError`, `BadRequestError`. Map provider-specific errors to these in each provider implementation.

4. **Capability-aware request building**  
   If model does not support `tools`, fall back to `add_function_to_prompt` style. If model does not support `response_format=json_schema`, fall back to JSON mode + client-side validation.

5. **Pre-call checks**  
   Estimate tokens and check against model context window before call. Drop/filter unsupported params per provider.

6. **Retry/fallback/cooldown**  
   Categorized fallbacks by error type, exponential backoff with jitter, cooldown model after repeated failures.

### C. Integration points

| hermes pattern | LiteLLM equivalent | LLMASM unified approach |
|---|---|---|
| `check_fn` availability | `get_supported_openai_params` | Use both: registry-level availability + provider-level capability checks |
| Tool schema sanitizer | `modify_params`, `drop_params` | Sanitize schemas and messages before provider call |
| Layered approval | Guardrails `pre_call` | Unified guardrail/middleware layer with pre/post hooks |
| Retry/fallback chain | Router fallbacks | `FallbackChain` in `RuntimeConfig` categorized by error type |
| Async bridging | — | Loop pool for async tools |
| Capability introspection | `supports_function_calling` | Provider protocol method |

---

## Part 4 — Proposed implementation phases

### Phase A: Tool registry refactor (hermes patterns)

1. Add `ToolEntry` wrapper and toolset abstraction.
2. Implement AST-based auto-discovery.
3. Add availability checks (`check_fn`) with TTL cache.
4. Add schema sanitizer and argument coercer.
5. Normalize `ToolRegistry.dispatch()` to JSON-string returns with error sanitization.
6. Add override protection and generation counter.

### Phase B: Tool safety & execution (hermes patterns)

1. Add layered approval/guardrail middleware.
2. Add untrusted result wrapping.
3. Add parallelism policy to scheduler.
4. Add resource limits/timeouts to tool nodes.
5. Add lifecycle hooks (`pre_tool_call`, `post_tool_call`).

### Phase C: Provider generalization (LiteLLM patterns)

1. Extend `LLMProvider` protocol with capability queries.
2. Add standardized exception taxonomy.
3. Implement `LiteLLMProvider`.
4. Refactor `OllamaProvider` as the `ollama/` implementation of the generalized protocol.
5. Add capability-aware request building (tools fallback, JSON schema fallback).

### Phase D: Reliability (LiteLLM + hermes)

1. Implement retry/fallback chain with categorized fallbacks.
2. Add pre-call context-window checks.
3. Add cooldown after repeated failures.
4. Add message integrity repair (orphaned tool calls/results).

---

## Part 5 — Configuration additions

Extend `RuntimeConfig` with tool/provider knobs:

```python
@dataclass
class RuntimeConfig:
    # Existing fields...

    # Toolsets
    enabled_toolsets: list[str] = field(default_factory=lambda: ["core"])
    tool_auto_discover: bool = True
    tool_availability_cache_ttl_seconds: float = 30.0

    # Tool execution
    tool_max_result_chars: int = 10000
    tool_default_timeout_seconds: float = 300.0
    tool_parallelism: str = "safe"  # "none" | "safe" | "all"

    # Approval / guardrails
    approval_mode: str = "interactive"  # "interactive" | "yolo" | "smart"
    guardrails: list[str] = field(default_factory=list)

    # Provider
    provider: str = "ollama"  # or "litellm", "openai", etc.
    fallback_providers: list[str] | None = None
    context_window_fallbacks: list[str] | None = None
    content_policy_fallbacks: list[str] | None = None
    num_retries: int = 3
    request_timeout_seconds: float = 120.0
    cooldown_after_fails: int = 3
    cooldown_seconds: float = 60.0
    drop_unsupported_params: bool = True
    modify_params: bool = True  # message sanitization
```

---

## Part 6 — Testing strategy

- Add unit tests in `tests/unit/` using `FakeProvider` and `InMemoryStorage`.
- New focused test files:
  - `test_tool_registry.py` — registration, toolsets, availability, override, generation counter.
  - `test_tool_dispatch.py` — coercion, sanitization, error normalization, middleware hooks.
  - `test_tool_approval.py` — hardline/pattern approval, env scrubbing.
  - `test_provider_capabilities.py` — capability queries, fallback chains.
  - `test_litellm_provider.py` — `LiteLLMProvider` integration (optional, may require mocks).
- Keep existing `tests/unit/test_v0.py` green.
- Run `make test`, `make lint`, and `make typecheck` after each phase.

---

## Part 7 — Open questions

1. **LiteLLM dependency:** Should LLMASM depend directly on `litellm`, or should `LiteLLMProvider` be an optional extra (`pip install llmasm[litellm]`)?
2. **Async tools:** Do you want to support async tool handlers now, or keep everything synchronous for v0?
3. **Approval UX:** Should approval be CLI-only for now, or do you want a pluggable backend interface from the start?
4. **Sandboxed code execution:** Is adding a `code_execution` tool in scope, or should we focus on the registry/dispatch safety patterns first?
5. **Tool auto-discovery:** Should discovery scan only `llmasm/tools/`, or also user plugin directories?
6. **Message integrity repair:** Should this run automatically before every model call, or be opt-in via `RuntimeConfig`?

---

## References

- `hermes-agent/tools/registry.py`
- `hermes-agent/toolsets.py`
- `hermes-agent/model_tools.py`
- `hermes-agent/agent/tool_executor.py`
- `hermes-agent/tools/approval.py`
- `hermes-agent/tools/code_execution_tool.py`
- `hermes-agent/tools/async_delegation.py`
- `llmasm/tools/registry.py`
- `llmasm/providers/base.py`
- `llmasm/providers/ollama.py`
- `llmasm/config.py`
- LiteLLM docs: function calling, exception mapping, fallbacks, guardrails, JSON mode.
