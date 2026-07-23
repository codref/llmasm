# LLMASM — LLM Assembly Scheduling Machine

Python 3.11+ library for local LLM orchestration with persistent graph memory.
Single package (not a monorepo); entrypoint is `llmasm/`.

## Commands

```bash
make test          # python -m pytest
make lint          # ruff check .
make typecheck     # mypy llmasm
```

Postgres integration tests require the e2e stack:

```bash
make e2e-stack-up  # docker compose up postgres + graph-viewer
LLMASM_TEST_DB=postgresql://llmasm:llmasm@localhost:15432/llmasm \
    python -m pytest tests/unit/test_postgres_storage.py -v
make e2e-stack-down
```

Tests are skipped automatically when `LLMASM_TEST_DB` is not set.
No CI workflows exist — these commands are the authoritative verification.

## Architecture

```
Prompt → Compiler → TaskGraph → Runtime VM → FinalAnswer
                  ↓                            ↓
            WorkspaceGraph ←── Persistence ───┘
```

- **`llmasm/api.py`** — Public facade class `LLMASM`. The `ask()` method chains compile + run + extract final answer. `compile()` and `run()` are available separately.
- **`llmasm/compiler/compiler.py`** — Prompt-to-graph compiler. Calls the planner model with structured output enforcement, validates the proposal, canonicalizes node/edge fields (e.g. injects `input_schema`/`output_schema` from tool specs when omitted), and has an automatic repair loop (`compiler_max_attempts`, default 5).
- **`llmasm/runtime/executor.py`** — VM-style executor with program counter, checkpoints, tool caching, and per-node retry/fatal handling. Only `intent`, `tool`, `model`, `compress`, `router`, `final`, and `expand` node kinds are implemented in v0; `memory_query`, `goal`, and `observation` raise `ExecutionError`.
- **`llmasm/runtime/scheduler.py`** — Computes the executable frontier. Propagates SKIPPED status for nodes on unselected router branches before picking the next PENDING node.
- **`llmasm/runtime/context.py`** — Context selection for model nodes: combines required upstream artifacts, graph neighborhood, active goal, and embedding-assisted memory retrieval.
- **`llmasm/graph/models.py`** — All Pydantic data models: `Node`, `TaskGraph`, `Run`, `Artifact`, `Checkpoint`, etc.
- **`llmasm/storage/base.py`** — `Storage` Protocol defining the persistence contract.
- **`llmasm/storage/memory.py`** — `InMemoryStorage` (dictionary-backed, for tests and simple examples).
- **`llmasm/storage/postgres.py`** — `PostgresStorage` (psycopg3, runs migrations on init).
- **`llmasm/storage/embeddings.py`** — `EmbeddingStore` Protocol; `InMemoryEmbeddingStore`, `NullEmbeddingStore`, helpers.
- **`llmasm/providers/base.py`** — `LLMProvider` base protocol (`generate`, `embed`, `list_models`).
- **`llmasm/providers/ollama.py`** — `OllamaProvider`, default URL `http://localhost:11434`.
- **`llmasm/tools/registry.py`** — `ToolRegistry`; tools must be registered before use. Tool specs reference schema names from the `SchemaRegistry`.
- **`llmasm/config.py`** — `RuntimeConfig` dataclass shared by compiler and runtime. Includes chunking/summary knobs for long source passages in fast-path chat.
- **`llmasm/chunking.py`** — `TextChunker` base class and `SentenceTextChunker`; used by the fast path to split long `source_passage` items into retrievable chunks.
- **`llmasm/tokenizers.py`** — `Tokenizer` abstract base class; concrete tokenizers can be plugged into `RuntimeConfig.tokenizer` and the chunker.

## Key rules for making changes

- **Storage is a Protocol, not a class.** Adding a method to `storage/base.py` means adding it to every implementation (`memory.py`, `postgres.py`).
- **Tools must be registered in `ToolRegistry`** before the compiler can use them. The registry validates that tool input/output schemas exist in `SchemaRegistry`.
- **Schema names are strings** resolved through `SchemaRegistry` → Pydantic model classes. Built-in schemas live in `llmasm/schemas.py`.
- **Node canonicalization happens in `compiler.py:_canonical_node_fields`.** The planner model may put schema names in `metadata` instead of top-level fields; the compiler normalizes both.
- **Final-node input schemas are reconciled deterministically** in `compiler.py:_reconcile_final_input_schemas` (called from `_normalize`). Final inputs are pinned to `Summary`; when a planner wires a mismatched, transform-less edge into a final node, model/compress sources are coerced to emit `Summary`, and for other sources the final node's input schema is adapted instead. This prevents weak planners from failing compilation with `SCHEMA_MISMATCH` (`RawText cannot connect to Summary`).
- **Orphaned tool outputs are wired into the answer chain.** `graph.validation.validate_tool_outputs_consumed` rejects tool nodes with no downstream consumer, and `compiler.py:_wire_orphaned_tools` (called from `_normalize`) recovers by adding a dedicated input port on the model that feeds the final node and connecting the tool output to it. This prevents weak planners from invoking a tool and then discarding its result.
- **Model-node prompts format tool outputs as natural language.** `runtime.executor.Executor._render_node_prompt` renders `RawText` as plain text and `WeatherObservation` as a human-readable weather line; other schemas fall back to JSON. When an input originates from a `tool` node, the instruction is augmented with a directive telling the model to base its answer on the tool result(s).
- **The planner prompt instructs clean entity extraction before tool calls.** Tools such as `weather.lookup`, `wikipedia.search`, `calculator.eval`, and `file.read` need concise inputs (city, topic, expression, path). The planner prompt tells the model to insert an extraction model node (`intent -> extract_entity -> tool -> model -> final`) instead of passing the full user sentence directly to the tool.
- **Embedding dimensions are locked per workspace.** Changing `embedding_dimensions` after a workspace has been created raises `StorageError`. Drop the vector column and re-initialize to switch models.
- **Long source passages are automatically chunked in fast-path chat.** When `chunking_enabled=True` and a `source_passage` exceeds `chunking_trigger_tokens`, the text is split into `source_passage` chunks and a summary node is run to produce a workspace-level summary. Chunks and summaries are both stored as `MemoryItem` objects with `metadata.is_summary` to distinguish them.

## Testing conventions

- All tests live in `tests/unit/`. No integration test directory.
- **`tests/unit/fakes.py`** is shared between tests and examples. Contains `FakeProvider` (deterministic LLM) and `ConversationRetrieveTool`. `FakeProvider(planner_outputs=[...])` consumes the list from the front on each structured-output call.
- **`InMemoryStorage`** is the standard test backend. It auto-collects few-shot examples from persisted task graphs.
- **Tool caching is on by default for tool nodes.** Re-executing a run with the same inputs will return the cached artifact without invoking the tool again (see `test_tool_cache_reuses_artifact_without_invocation`). To bypass, set `allow_cache: false` in the node's execution dict.
- Postgres tests skip unless `LLMASM_TEST_DB` is set in the environment.

## Docker / services

- Postgres on port 15432, graph viewer on port 3000 (`docker compose up`).
- Postgres image includes Apache AGE extension (build arg `AGE_REF`).
- `pip install -e ".[dev]"` for development deps (pytest, ruff, mypy).
