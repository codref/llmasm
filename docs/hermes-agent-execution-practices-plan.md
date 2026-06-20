# Plan: Import hermes-agent execution practices into LLMASM

**Status:** Draft plan — not yet implemented  
**Goal:** Strengthen LLMASM's executor, context builder, compiler, persistence, and configuration with concrete, proven practices from `hermes-agent`, while keeping LLMASM's graph-first architecture intact.

---

## Background

`hermes-agent` is a mature, production-oriented personal agent (v0.16.0) built around a synchronous tool-calling conversation loop. LLMASM is a v0.1.0 research library built around a persistent typed graph and a VM-style executor. Although the architectures differ, many of hermes-agent's runtime hardening techniques map cleanly into LLMASM's components.

This plan identifies the highest-value practices and proposes an implementation order.

---

## Architecture mapping

| hermes-agent component | LLMASM component | What to import |
|---|---|---|
| `agent/conversation_loop.py` | `llmasm/runtime/executor.py` | Budgets, retries, interrupts, streaming |
| `agent/turn_context.py` | `llmasm/compiler/compiler.py`, `llmasm/runtime/executor.py` | Per-turn context prologue, early persistence |
| `agent/context_compressor.py` | `llmasm/runtime/context.py` | Proactive compression, summary reuse |
| `model_tools.py`, `tools/registry.py` | `llmasm/tools/registry.py`, `llmasm/runtime/executor.py` | Argument coercion, error sanitization |
| `hermes_state.py` | `llmasm/storage/` | Schema reconciliation, WAL patterns, jittered retries |
| CLI / `.env` config | `llmasm/config.py` | Runtime knobs |

---

## Phase 1 — Foundation (quick wins)

### 1.1 Add runtime knobs to `RuntimeConfig`

Extend `llmasm/config.py` with:

- `max_iterations: int = 32`
- `budget_grace_call: bool = True`
- `api_max_retries: int = 3`
- `api_retry_base_seconds: float = 5.0`
- `api_retry_cap_seconds: float = 120.0`
- `fallback_providers: list[str] | None = None`
- `streaming_default: bool = False`
- `compression_enabled: bool = False`
- `compression_threshold_tokens: int = 4096`
- `summary_model: str | None = None`
- `prompt_cache_breakpoints: bool = False`

These knobs make later features configurable and testable without hard-coded constants.

### 1.2 Turn-level iteration budget with grace call

Add budget state to `Run` (or `Executor`) and gate `Executor.execute()`:

- Decrement budget before every model/tool/router/compress node.
- When budget reaches zero, allow one final `final`/`model` node to produce a closing answer if `budget_grace_call` is true.
- Emit a `NodeStatus.failed` with a clear `ExecutionError` message if the budget is exhausted and no graceful close is possible.

### 1.3 Tool argument coercion and error sanitization

Before `tool.invoke()` in `executor.py`:

- Coerce string arguments to schema-declared types (`"42"` → `int`, `"true"` → `bool`, bare scalar → `list` when the schema expects an array).
- Sanitize tool errors by stripping XML role tags, markdown code fences, CDATA, and capping length before persisting as an artifact.

This protects against weak planner output and malformed tool responses.

---

## Phase 2 — Context discipline and prompt hygiene

### 2.1 Stable system-prompt snapshot

- Render the system prompt once per `Run`, persist it as an artifact, and restore it on continuation/follow-up.
- Strip internal keys (`finish_reason`, `_thinking_prefill`) before sending to the provider.
- Normalize whitespace and sort tool JSON deterministically to keep provider prefix caches warm.
- Add optional cache-control breakpoints when `prompt_cache_breakpoints` is enabled.

### 2.2 Proactive context compression

In `llmasm/runtime/context.py`:

- Estimate token count for the assembled context before each model/compress node.
- If over `compression_threshold_tokens`, compress:
  1. Protect system prompt + first N turns + latest assistant/user tail.
  2. Summarize the middle with a dedicated `compress` node.
  3. Store the summary as a `MemoryItem` with `metadata.is_summary`.
  4. Reuse the previous summary on subsequent compressions instead of re-summarizing from scratch.

This directly uses LLMASM's existing `MemoryItem` and chunking infrastructure.

### 2.3 Pre-flight role/message integrity repair

Before execution, validate the assembled context:

- Drop orphaned tool results without matching tool calls.
- Insert stub results for tool calls without matching results.
- Ensure user/assistant/tool roles alternate correctly.

This can be implemented as a small repair pass inside `Executor.execute()` or as part of context assembly.

---

## Phase 3 — Robust provider interaction

### 3.1 Per-call retry/fallback chain

Wrap `provider.generate()` calls in `executor.py`:

- Exponential backoff with jitter between `api_retry_base_seconds` and `api_retry_cap_seconds`.
- Classify errors: retry on transport/rate-limit/context-length/malformed-empty responses; treat content-policy refusals as terminal.
- If `fallback_providers` is set, switch provider after retries are exhausted.

Introduce a `RetryableError` vs `FatalProviderError` distinction in `llmasm/providers/base.py` or a new `llmasm/errors.py` module.

### 3.2 Interrupt handling and `/steer` drain

- Add an interrupt flag to `Executor`/`Run` that `Executor.execute()` checks between nodes.
- On interrupt, stop cleanly and persist current state.
- Add a `/steer` queue: user mid-thought notes are appended to the next tool node's context without breaking role alternation.

### 3.3 Streaming execution path

- Extend `LLMProvider` with a streaming `generate` method returning deltas.
- Consume deltas inside `Executor` for model/compress/router nodes.
- Use streaming even when no consumer is attached, to detect stale connections and enable interrupts.

This is the largest change and should be done after the retry/fallback work is stable.

---

## Phase 4 — Persistence hardening

### 4.1 Jittered write retries

- Wrap `PostgresStorage` write operations (and any future SQLite session store) with short, jittered retries.
- This is relevant for concurrent `Run` updates and checkpoint writes.

### 4.2 Declarative schema reconciliation (Postgres)

- On `PostgresStorage` initialization, compare live table columns against canonical DDL.
- `ALTER TABLE ADD COLUMN` for any missing columns automatically.
- Keep migrations in `llmasm/storage/migrations/` as the canonical source of truth.

If a SQLite session store is added later, reuse the reconciliation logic and add WAL fallback.

---

## Phase 5 — Compiler/turn hygiene

### 5.1 `TurnContext` prologue

Introduce a `TurnContext` dataclass built at the start of `LLMASM.ask()` / `LLMASM.chat()` / `Executor.execute()`:

- Sanitized user message
- Run/task IDs
- Active system prompt snapshot
- Reset retry/budget counters
- Plugin/prefetch context (when plugins exist)

Persist the inbound `Run` state before entering the node loop so crashes do not lose the user request.

---

## Proposed implementation order

1. **Config knobs** (`RuntimeConfig`) — unblocks everything else.
2. **Iteration budget + grace call** — high impact, low risk.
3. **Tool argument coercion + error sanitization** — high impact, low risk.
4. **`TurnContext` prologue + early persistence** — cleaner orchestrator.
5. **Stable system-prompt snapshot** — cost/performance win.
6. **Retry/fallback chain** — production hardening.
7. **Pre-flight role/message integrity repair** — fixes weak planner edge cases.
8. **Proactive context compression** — scalability.
9. **Interrupt handling + `/steer` drain** — better UX.
10. **Streaming execution path** — largest change, do last.
11. **Persistence: jittered retries + schema reconciliation** — harden Postgres backend.

---

## Testing strategy

- Add unit tests in `tests/unit/` using `FakeProvider` and `InMemoryStorage`.
- For each practice, add at least one behavioral test:
  - Budget exhaustion produces a graceful final answer or a clear error.
  - String tool args are coerced to declared types.
  - Retry loop triggers on `FakeProvider` raising retryable errors.
  - Context compression stores a `MemoryItem` with `is_summary=true`.
- Keep existing `tests/unit/test_v0.py` green; add new focused files (e.g., `test_executor_budget.py`, `test_context_compression.py`) rather than bloating the main test file.
- Run `make test`, `make lint`, and `make typecheck` after each phase.

---

## Open questions

1. Should streaming be implemented as a required `LLMProvider` method, or as an optional capability with a non-streaming fallback?
2. Should we add a SQLite session store alongside Postgres, or keep Postgres as the only persistence backend?
3. Should context compression run synchronously inside the executor, or be emitted as explicit `compress` nodes in the graph?
4. Should fallback providers be selected globally in `RuntimeConfig`, or per-node/per-run?

---

## References

- `hermes-agent/agent/conversation_loop.py` — loop control, retries, streaming
- `hermes-agent/agent/turn_context.py` — per-turn context prologue
- `hermes-agent/agent/context_compressor.py` — compression and integrity repair
- `hermes-agent/model_tools.py` — tool dispatch and sanitization
- `hermes-agent/tools/registry.py` — tool registry and caching
- `hermes-agent/hermes_state.py` — SQLite session persistence
- `llmasm/runtime/executor.py` — current VM executor
- `llmasm/runtime/context.py` — current context selection
- `llmasm/compiler/compiler.py` — current compiler/repair loop
- `llmasm/config.py` — current runtime configuration
