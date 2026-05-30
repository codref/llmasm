# LLMASM Implementation Plan

Status: **In Progress — V0 near-complete**

| Legend | Meaning |
|--------|---------|
| ✅ | Implemented and tests passing |
| ⚠️ | Implemented with known gap — see note |
| ⏸ | Deferred |
| 🔲 | Not started |

### Current status (updated 2026-05-30)

All phases through 14 are complete. Phase 12 gaps closed this session:
- `RuntimeConfig.embedding_dimensions: int = 768` added
- `002_pgvector.sql` refactored — column DDL moved to `PostgresEmbeddingStore.__init__`
- `PostgresEmbeddingStore` is now dimension-aware with a startup guard against mismatches
- `select_context` vector path implemented (was a stub); merged with word-overlap, deduped by item id
- `executor.py` passes `provider` to `select_context`
- `examples/chat.py` wires `--embeddings` / `--embedding-dimensions` flags
- Test fixtures for embedding DDL teardown fixed to prevent column-leak between sessions

Only Phase 15 (Apache AGE) remains deferred.

---

Purpose: phased build plan for implementing LLMASM from the RFC in `docs/llmgraph-rfc.md`. This document is written for code-generation agents and contributors, so each phase has concrete deliverables, module boundaries, and acceptance checks.

## Guiding Constraints

- Package name: `llmasm`.
- Python target: Python 3.11+.
- Default implementation path: plain Postgres-compatible schema first, with Apache AGE and pgvector integration added behind interfaces.
- Local LLM backend: Ollama first.
- V0 execution model: single-process, synchronous executor. Async and distributed execution are later extensions.
- Keep external dependencies minimal: prefer stdlib, `pydantic`, `psycopg`, and `httpx` unless a phase explicitly requires more.
- All public objects should be typed and documented with docstrings.
- Tests should use in-memory fakes where possible; integration tests that require Postgres or Ollama must be marked separately.

## Target Repository Shape

```text
llmasm/
  __init__.py
  config.py
  ids.py
  errors.py
  schemas.py
  graph/
    models.py
    registry.py
    validation.py
    transforms.py
  goals/
    classifier.py
    tracker.py
  tools/
    base.py
    registry.py
  providers/
    base.py
    ollama.py
  compiler/
    proposal.py
    prompt.py
    parser.py
    validators.py
    repair.py
    compiler.py
  runtime/
    scheduler.py
    context.py
    executor.py
    expansion.py
  storage/
    base.py
    embeddings.py
    memory.py
    postgres.py
    migrate.py
    migrations/
  analysis/
    run.py
tests/
  unit/
  integration/
docs/
  llmgraph-rfc.md
  llmasm-implementation-plan.md
```

## ✅ Phase 0: Project Skeleton And Tooling

Goal: create a runnable Python package with test, lint, and packaging basics.

Implementation tasks:

- Add `pyproject.toml` with package metadata for `llmasm`.
- Configure dependencies:
  - runtime: `pydantic`, `psycopg[binary]`, `httpx`;
  - dev: `pytest`, `pytest-cov`, `ruff`, `mypy`.
- Add `llmasm/__init__.py` exporting the version only.
- Add `llmasm/errors.py` with base exception classes:
  - `LLMASMError`;
  - `ValidationError`;
  - `CompilationError`;
  - `ExecutionError` — base class for all execution-time failures;
  - `RetryableError(ExecutionError)` — raised by node handlers when the failure is transient and the executor should apply the node's retry policy (e.g. model timeout, tool rate limit);
  - `FatalError(ExecutionError)` — raised by node handlers when the failure is unrecoverable for the current run; the executor marks the run `failed` immediately;
  - `StorageError`;
  - `ProviderError`.
- Add `llmasm/ids.py` with deterministic helper functions for prefixed IDs:
  - `new_id(prefix: str) -> str`;
  - accepted prefixes: `workspace`, `taskgraph`, `run`, `node`, `edge`, `artifact`, `goal`, `memory`, `checkpoint`.
- Add CI-friendly commands in project docs or Makefile:
  - `pytest`;
  - `ruff check .`;
  - `mypy llmasm`.

Acceptance criteria:

- `python -c "import llmasm"` succeeds.
- `pytest` runs with at least one smoke test.
- No production code talks to Ollama or Postgres yet.

## ✅ Phase 1: Core Domain Models

Goal: define the in-memory data structures used by every later phase.

Implementation tasks:

- Add `llmasm/graph/models.py` with Pydantic models:
  - `WorkspaceGraph`;
  - `TaskGraph`;
  - `Node`;
  - `Port`;
  - `TaskEdge`;
  - `WorkspaceEdge`;
  - `Run`;
  - `RunNodeState`;
  - `Artifact`;
  - `ToolCall`;
  - `ModelCall`;
  - `MemoryItem`;
  - `EmbeddingRef(id: str, owner_type: str, owner_id: str, model: str, dimensions: int, text_hash: str, created_at: datetime)` — carries the identity and provenance of one embedding computation. No vector field: the float vector is stored separately through the `EmbeddingStore` protocol introduced in Phase 12. `owner_type` is one of `artifact`, `memory_item`, `node`, `prompt`. `text_hash` is a SHA-256 hex digest of the input text, used for deduplication;
  - `Checkpoint`;
  - `Goal`.
- `Artifact` must include a `superseded_by: str | None` field. When a node is re-executed, the new artifact is written as a new record and the prior artifact's `superseded_by` is set to the new artifact's ID. Artifacts remain append-only; `superseded_by` is the only mutable field. `TaskEdge` and `WorkspaceEdge` are the two concrete edge types used throughout the plan. `TaskEdge` carries execution-scoped dataflow with ports and optional transforms, matching the `task_edges` table in the RFC §6 storage schema. `WorkspaceEdge` carries semantic and provenance links across prompts, task graphs, artifacts, goals, and memory items, matching `workspace_edges`. Both derive from the single conceptual `Edge` in RFC §4; the split follows RFC §6 because port/transform fields are only meaningful within a task subgraph.
- Use enums for stable fields:
  - `NodeKind`: `intent`, `tool`, `model`, `memory_query`, `compress`, `router`, `expand`, `goal`, `observation`, `final`;
  - `RunStatus`: `pending`, `running`, `succeeded`, `failed`, `cancelled`;
  - `NodeStatus`: `pending`, `running`, `succeeded`, `failed`, `retryable`, `expanded`, `skipped`;
  - `GoalAction`: `continue`, `steer`, `new`;
  - `WorkspaceEdgeType`: `depends_on`, `produced`, `used_context`, `summarizes`, `refers_to`, `follows_up`, `supports_goal`, `contradicts`, `expands_to`. `sets_or_updates` appears in the RFC §5 follow-up diagram for a goal assignment event but is not in the RFC §4 edge taxonomy. It is **not included** in this enum: the compiler should emit `supports_goal` when a prompt or task graph updates the active goal, which covers the same semantic intent without a redundant edge type.
- Ensure `Node` has no runtime `status` field. Runtime state belongs only to `RunNodeState`.
- Add convenience constructors for common nodes only if they reduce repeated test setup.

Acceptance criteria:

- Unit tests can instantiate every model with valid sample data.
- Invalid enum values fail validation.
- A `TaskGraph` can be represented without execution state.
- A `Run` can reference a `TaskGraph` and maintain independent `RunNodeState` records.

## ✅ Phase 2: Schema Registry And Transforms

Goal: support typed ports and deterministic edge compatibility.

Implementation tasks:

- Add `llmasm/schemas.py` with built-in Pydantic data models:
  - `ConversationRecord(id: str, text: str, metadata: dict)`;
  - `ConversationText(text: str)`;
  - `Summary(text: str, source_id: str | None)`;
  - `WeatherQuery(location: str | None, date: str | None)`;
  - `WeatherObservation(condition: str, source_url: str | None)`;
  - `FinalAnswer(text: str, sources: list[str])`;
  - `RawText(text: str)`;
  - `JsonValue(value: dict | list | str | int | float | bool | None)`;
  - `NotFound(resource_type: str, resource_id: str, detail: str)` — returned by tool nodes when a requested resource does not exist (RFC §13 "Missing Conversation"); tool nodes that return this type must route to a final response node instead of propagating to downstream model nodes.
- Add `llmasm/graph/registry.py`:
  - `SchemaRegistry.register(tag: str, model: type[BaseModel])`;
  - `SchemaRegistry.get(tag: str)`;
  - `SchemaRegistry.has(tag: str)`;
  - `SchemaRegistry.describe() -> str`;
  - `default_schema_registry()`.
- Add `llmasm/graph/transforms.py`:
  - `TransformRegistry.register(name, from_schema, to_schema, fn)`;
  - `TransformRegistry.resolve(name)`;
  - `TransformRegistry.can_transform(from_schema, to_schema, name)`;
  - built-ins: `extract_text`, `to_json_string`, `select_field`.
- Keep `select_field` represented as a parameterized transform object or a transform name plus metadata. Do not encode arbitrary Python expressions in planner output.

Acceptance criteria:

- Direct schema equality is accepted.
- Registered transforms bridge compatible unequal schemas.
- Unknown schema tags and transforms produce deterministic validation errors.
- `extract_text` converts `ConversationRecord` to `ConversationText`.

## ✅ Phase 3: Tool Registry And Provider Interfaces

Goal: define invocation boundaries without building the compiler or executor yet.

Implementation tasks:

- Add `llmasm/tools/base.py`:
  - `ToolSpec(name, description, input_schema, output_schema)`;
  - `Tool` protocol with `spec() -> ToolSpec` and `invoke(input: BaseModel) -> BaseModel`.
- Add `llmasm/tools/registry.py`:
  - register tool;
  - retrieve by name;
  - describe tools for planner prompt;
  - validate input/output schema tags.
- Add `llmasm/providers/base.py`:
  - `ModelInfo(name: str, context_window: int | None)`;
  - `ModelOutput(text: str, raw: dict | None, token_usage: dict | None)`;
  - `EmbeddingOutput(vector: list[float], raw: dict | None)`;
  - `LLMProvider` protocol with `name`, `list_models`, `generate`, and `embed`;
  - `TokenizerProtocol` with a single method `count_tokens(text: str) -> int`; the default implementation uses `ceil(len(text.split()) * 1.35)` and can be substituted in `RuntimeConfig` for model-specific tokenizers.
- Add `llmasm/providers/ollama.py`:
  - local HTTP implementation using `httpx`;
  - configurable base URL, timeout, model defaults;
  - `generate(prompt, options, format_schema: dict | None = None)` — `format_schema` is a JSON Schema dict produced by `TaskGraphProposal.model_json_schema()` and passed to Ollama's `format` field to enforce structured output;
  - `embed(texts, options)`.
- Add a fake provider for tests under `tests/unit/fakes.py`.

Acceptance criteria:

- Tool registry can render deterministic prompt descriptions.
- Fake tool invocation validates input and output types.
- Ollama provider is importable but integration tests are skipped unless `OLLAMA_BASE_URL` is set.

## ✅ Phase 4: Storage Interfaces And In-Memory Store

Goal: build against a storage abstraction before introducing Postgres complexity.

Implementation tasks:

- Add `llmasm/storage/base.py` with `Storage` protocol methods:
  - workspace graph: create/load;
  - task graph: persist/load;
  - runs: create/load/update;
  - run node states: create/list/update;
  - task edges and workspace edges: persist/list/traverse minimal;
  - artifacts: persist/load/list;
  - goals: load active/create provisional/finalize/steer;
  - checkpoints: persist/list;
  - tool/model calls: persist/list;
  - compilation failures: persist;
  - `retrieve_few_shot_examples(workspace_graph_id, intent, limit) -> list[FewShotExample]`;
  - `find_cached_artifact(node_execution_key, input_artifact_ids) -> Artifact | None`.
- Embedding persistence and vector search are **not** part of the `Storage` protocol. They belong to the separate `EmbeddingStore` protocol in `llmasm/storage/embeddings.py` (Phase 12). This separation keeps the main storage contract testable without a vector database.
- Add `llmasm/storage/memory.py` implementing the protocol with dictionaries.
- Model all persisted writes as immutable where the RFC requires it:
  - artifacts are append-only;
  - checkpoints are append-only;
  - model/tool calls are append-only;
  - run and run-node-state statuses are mutable.
- Implement `load_workspace_edges_for_task(task_graph_id)` by finding workspace edges connected to the task graph, its nodes, its goal, or its root prompt.

Acceptance criteria:

- Unit tests can create a workspace, task graph, run, node states, artifacts, and checkpoints through the storage interface.
- A task graph can be executed twice with separate run node states.
- Workspace edges can link heterogeneous endpoint types.

## ✅ Phase 5: Graph Validation

Goal: validate task graphs deterministically before execution.

Implementation tasks:

- Add `llmasm/graph/validation.py` with validation functions:
  - `validate_required_ports`;
  - `validate_schema_refs`;
  - `validate_edge_compatibility`;
  - `validate_tools`;
  - `validate_models`;
  - `validate_context_budgets`;
  - `validate_terminal_node`;
  - `validate_acyclic`;
  - `validate_workspace_links`.
- Use a structured error model:
  - `ValidationIssue(code: str, node_name: str | None, detail: str)`.
- Note: `ValidationIssue` is a structured result record, not an exception. The exception class defined in `errors.py` is `ValidationError`. These are intentionally different: `ValidationIssue` is returned in lists from pure validator functions; `ValidationError` is raised as an exception only when a hard schema boundary is violated outside the compiler path (e.g., a caller passes an unknown schema tag directly to the registry).
- Implement exact error codes from the RFC:
  - `PORT_UNSATISFIED`;
  - `SCHEMA_MISMATCH`;
  - `UNKNOWN_TOOL`;
  - `UNKNOWN_MODEL`;
  - `CONTEXT_BUDGET_EXCEEDED`;
  - `UNKNOWN_SCHEMA`;
  - `UNKNOWN_TRANSFORM`;
  - `NO_TERMINAL_NODE`;
  - `ILLEGAL_CYCLE`;
  - `MISSING_WORKSPACE_LINK`;
  - `INVALID_WORKSPACE_TARGET`;
  - `GOAL_ACTION_MISMATCH`.
- Keep validation pure: it should not mutate storage.

Acceptance criteria:

- Each validation error code has at least one unit test.
- A valid conversation-summary task graph passes.
- A follow-up task graph with `continue` and no workspace link fails.
- A planner proposal whose `goal_action` differs from expected fails.

## ✅ Phase 6: Goal Tracking

Goal: implement deterministic goal classification and lifecycle management.

Implementation tasks:

- Add `llmasm/goals/classifier.py`:
  - `classify_goal_action(prompt: str, active_goal: Goal | None) -> GoalAction`;
  - implement the RFC keyword and word-overlap heuristic.
- Add `llmasm/goals/tracker.py`:
  - `load_active_goal(workspace_graph_id)`;
  - `create_provisional_goal(workspace_graph_id, prompt)`;
  - `finalize_goal(goal_id, goal_update_text)`;
  - `steer_goal(goal_id, goal_update_text)`;
  - `close_goal(goal_id, reason)`;
  - this layer may be a thin wrapper over storage.
- For `new`, create provisional goal before planner call and finalize after accepted proposal.
- For `steer`, update the existing active goal after accepted proposal.
- For `continue`, do not change goal text.

Acceptance criteria:

- No active goal classifies as `new`.
- `"now check if..."` classifies as `continue`.
- `"actually..."` classifies as `steer`.
- `"new task..."` classifies as `new`.
- Provisional goals are finalized only after proposal validation succeeds.

## ✅ Phase 7: Compiler Proposal Models And Prompt Renderer

Goal: produce and parse structured task graph proposals.

Implementation tasks:

- Add `llmasm/compiler/proposal.py`:
  - Pydantic models for `TaskGraphProposal`, `ProposalNode`, `ProposalPort`, `ProposalEdge`, `WorkspaceLinkProposal`, `ProposalExecution`.
  - Enforce structural constraints that Pydantic can handle directly.
- Add `llmasm/compiler/parser.py`:
  - `parse_task_graph_proposal(raw: str) -> ParseResult`;
  - return parse errors as `ValidationIssue(code="PARSE_FAILURE", ...)`.
- Add `llmasm/compiler/prompt.py`:
  - render fixed planner prompt from schema registry, tool descriptions, model list, active goal, prior context, user prompt;
  - delegate token counting to a `TokenizerProtocol` instance from `RuntimeConfig`; default implementation uses `ceil(len(text.split()) * 1.35)`;
  - subtract `repair_section_reserve` from total budget before context retrieval;
  - trim context items before required sections;
  - implement Section 7 few-shot retrieval: after sections 1–6 are budgeted, retrieve up to 2 prior `TaskGraphProposal` JSON examples from successfully compiled task graphs whose `intent` is semantically similar to the current prompt. Use text-overlap ranking in v0 (no embeddings required yet). Include both examples only when budget allows; drop the second then the first if space is tight. Store retrieved examples in `FewShotExample(proposal_json: str, intent: str, task_graph_id: str)` value objects. Add `retrieve_few_shot_examples(workspace_graph_id, intent, limit) -> list[FewShotExample]` to the `Storage` protocol (in-memory implementation returns an empty list until prior compilations exist).
- Define `RuntimeConfig` in `llmasm/config.py`:
  - `planner_model`;
  - `planner_max_tokens = 6144`;
  - `compiler_max_attempts = 3`;
  - `repair_section_reserve = 300` (tokens reserved in the planner budget for the error injection section on retry attempts);
  - `tokenizer: TokenizerProtocol` (defaults to the word-split estimator);
  - `scope`;
  - `default_model`;
  - `default_context_tokens`;
  - `embedding_model: str = "nomic-embed-text"` — model name passed to `provider.embed()`; must match a model available in the configured Ollama instance;
  - `embeddings_enabled: bool = False` — opt-in flag; when `False`, the embedding write path and vector search are skipped entirely and the context selector falls back to text-overlap ranking; set to `True` only when the `EmbeddingStore` backend and the embedding model are confirmed available.

Acceptance criteria:

- Prompt rendering is deterministic for the same inputs.
- Context items are trimmed when budget is too small.
- Invalid JSON returns `PARSE_FAILURE`.
- A valid JSON proposal parses into typed proposal objects.

## ✅ Phase 8: Compiler Repair Loop And Normalization

Goal: compile a user prompt into a persisted task graph.

Implementation tasks:

- Add `llmasm/compiler/repair.py`:
  - `compile_with_repair(planner, planner_prompt, expected_goal_action, max_attempts)`;
  - call provider with structured output;
  - parse output;
  - run structural and semantic validators;
  - append validation errors to retry prompt;
  - raise `CompilationError` with attempts, last errors, and last raw output.
- Add `llmasm/compiler/compiler.py`:
  - `compile_into_workspace(workspace_graph_id, prompt, runtime_config) -> task_graph_id`;
  - load workspace;
  - classify goal;
  - create provisional goal for `new`;
  - retrieve prior context from storage or memory search stub;
  - render planner prompt;
  - run repair loop;
  - normalize proposal into `TaskGraph`, `Node`, `Port`, `TaskEdge`, and `WorkspaceEdge` objects;
  - finalize or steer goal based on accepted proposal;
  - persist task graph before execution.
- Normalization rules:
  - generate IDs for all proposed nodes and edges;
  - preserve proposal node names in metadata for debugging;
  - convert `workspace_links` into `WorkspaceEdge` rows;
  - create a root prompt node if the proposal did not include one.

Acceptance criteria:

- Mock planner returning one invalid then one valid proposal succeeds after repair.
- Mock planner returning mismatched `goal_action` fails with `GOAL_ACTION_MISMATCH`.
- Compilation failure is persisted.
- Successful compilation persists task graph, task edges, workspace edges, and goal link.

## ✅ Phase 9: Scheduler And Runtime Executor

Goal: execute a validated task graph through a VM-style loop.

Implementation tasks:

- Add `llmasm/runtime/scheduler.py`:
  - compute executable frontier from task edges and run node states;
  - `next_node(run) -> Node | None`;
  - skip already succeeded/expanded nodes;
  - terminal condition: no pending executable nodes and at least one final node succeeded.
- Add `llmasm/runtime/context.py`:
  - `select_context(run, node, direct_inputs, embedding_store: EmbeddingStore | None)` — `| None` is intentional here: Phase 9 is built before `EmbeddingStore` exists in Phase 12, so `None` is the valid sentinel for that phase. Phase 12c tightens the signature to `EmbeddingStore` (non-optional) once `NullEmbeddingStore` is available as the disabled-path substitute;
  - include required direct inputs, active goal, node instruction;
  - retrieve graph candidates through storage traversal (always active);
  - when `embedding_store` is not `None` and `runtime_config.embeddings_enabled` is `True`: embed the node's intent text, call `embedding_store.search_similar(query_vector, filters, limit=20)`, and merge results with graph candidates before ranking;
  - when `embedding_store` is `None` or embeddings are disabled: retrieve memory candidates by text-overlap score against memory items — this is the v0 default path, not a temporary stub;
  - rank by required first, then relevance score, then token cost;
  - never include full prior run history by default.
- Add `llmasm/runtime/executor.py`:
  - `execute(run_id) -> Run`;
  - create initial `RunNodeState` rows when run starts;
  - update `program_counter`;
  - persist checkpoint before and after each node;
  - before invoking any node, call `storage.find_cached_artifact(node_execution_key, input_artifact_ids)`; if a non-superseded match is found and the node's `execution.allow_cache` is `True`, skip invocation and reuse the cached artifact directly (`allow_cache` defaults to `True` for `tool` nodes, `False` for `model` nodes);
  - gather inputs from upstream artifacts and transforms;
  - invoke tool/model/final/expand nodes;
  - persist artifacts, tool calls, model calls;
  - mark node states succeeded, failed, retryable, expanded, or skipped.
- Implement v0 node handlers:
  - `intent`: creates artifact from literals/metadata;
  - `tool`: invokes registered tool; if output is `NotFound`, mark node succeeded and set a `not_found` flag on `RunNodeState.metadata` so the scheduler can skip downstream model nodes;
  - `model`: calls provider;
  - `compress`: calls provider with a summarization prompt template; reduces the upstream artifact to a token-budget-respecting summary artifact. Required for context budget management (RFC §12). Use the node's `execution.prompt_template` field (default: `compress_to_summary.v1`) and `execution.max_input_tokens` as the output token target;
  - `final`: assembles final answer from upstream artifact;
  - `expand`: expects an `ExpansionRequest`;
  - `memory_query`, `router`, `goal`, `observation` are not implemented in v0; encountering one raises `ExecutionError` with a message identifying the kind.

Acceptance criteria:

- Conversation-summary task graph executes with fake retriever and fake model.
- Only retriever output reaches summarizer in selected context.
- Every executed node creates node state updates and checkpoints.
- Tool and model calls are persisted.
- Missing required input fails the node deterministically.
- Executing a node with an unsupported kind raises `ExecutionError` with a message that identifies the kind name.
- A tool node with matching inputs and `allow_cache: True` returns a cached artifact without re-invoking the tool.
- A model node with `allow_cache: False` always re-invokes the provider even when a matching artifact exists.

## ✅ Phase 10: Runtime Expansion

Goal: allow ReAct-style reasoning nodes to inject validated work.

Implementation tasks:

- Add `llmasm/runtime/expansion.py`:
  - `ExpansionRequest`;
  - `Expansion`;
  - `validate_expansion(run, source_node, request)`.
- Validation checks:
  - maximum proposed nodes;
  - known node kinds;
  - known tools;
  - valid ports;
  - schema compatibility;
  - no illegal cycles;
  - relevance reason is non-empty;
  - provenance edge `expands_to` is created from source node.
- Apply expansion:
  - persist new nodes;
  - persist new task edges;
  - persist workspace provenance edges;
  - initialize `RunNodeState` rows for new nodes in the current run;
  - let scheduler see new nodes in the next loop.

Acceptance criteria:

- A valid weather lookup expansion adds a tool node and executes it.
- Unknown tool expansion is rejected.
- Cycle-producing expansion is rejected.
- Expansion request and created IDs are persisted.

## ✅ Phase 11: Postgres Persistence

Goal: replace in-memory storage with a plain Postgres backend matching the RFC fallback schema.

Implementation tasks:

- Add `llmasm/storage/migrations/001_initial.sql` with tables:
  - `schema_version(version integer primary key, applied_at timestamptz not null)` — tracks applied migrations;
  - `embeddings(id text primary key, owner_type text not null, owner_id text not null, model text not null, dimensions integer not null, text_hash text not null, vector_json jsonb, created_at timestamptz not null)` — `vector_json` stores the float array as JSON when pgvector is unavailable; a later migration adds a native `vector(N)` column when the extension is present. Index on `(owner_type, owner_id)` for fast lookup by owner.
- Add `llmasm/storage/migrate.py`:
  - `run_migrations(conn) -> int` — applies any unapplied migration files in `migrations/` in numeric order; returns the number of migrations applied; idempotent if schema is already current;
  - migration files are named `NNN_<description>.sql`; the runner reads `schema_version`, skips already-applied versions, applies remaining files in order inside individual transactions, and records each applied version;
  - the runner is invoked explicitly (e.g. `llmasm migrate` CLI or `storage.run_migrations()` in integration test setup); it is never called automatically at import time.
- Add remaining tables:
  - `workspace_graphs`;
  - `task_graphs`;
  - `runs`;
  - `nodes`;
  - `task_edges`;
  - `workspace_edges`;
  - `run_node_states`;
  - `artifacts`;
  - `tool_calls`;
  - `model_calls`;
  - `expansion_requests`;
  - `goals`;
  - `memory_items`;
  - `embeddings`;
  - `checkpoints`;
  - `compilation_failures`.
- Add indexes:
  - by `workspace_graph_id`;
  - by `task_graph_id`;
  - by `run_id`;
  - by `node_id`;
  - by `workspace_edges(from_type, from_id)`;
  - by `workspace_edges(to_type, to_id)`;
  - by `run_node_states(run_id, status)`.
- Add primary key constraint on `run_node_states(run_id, node_id)` to enforce that a node appears exactly once per run.
- Add `llmasm/storage/postgres.py` implementing `Storage`.
- Store JSON payloads in `jsonb`.
- Keep migrations explicit. Do not auto-mutate schemas at import time.

Acceptance criteria:

- `run_migrations()` on a fresh database applies `001_initial.sql` and records version 1 in `schema_version`.
- Running `run_migrations()` a second time applies zero migrations and returns 0.
- Postgres integration tests can apply migration, run compiler with fake planner, execute fake graph, and query run analysis.
- In-memory and Postgres storage pass the same storage contract tests.
- Artifacts and checkpoints are append-only.

## ✅ Phase 12: Memory, Embeddings, And Context Wiring

Goal: replace the text-overlap stub in the context selector with a real retrieval layer; add an embedding store abstraction testable without pgvector.

This phase has four independent sub-goals that can be implemented in order:

### ✅ 12a — EmbeddingStore abstraction

- Add `llmasm/storage/embeddings.py`:
  - `ScoredMatch(item: MemoryItem | Artifact, score: float, embedding_id: str)`;
  - `EmbeddingStore` protocol:
    - `persist(ref: EmbeddingRef, vector: list[float]) -> None`;
    - `search_similar(query_vector: list[float], filters: dict, limit: int) -> list[ScoredMatch]`;
    - `find_by_owner(owner_type: str, owner_id: str) -> EmbeddingRef | None`;
    - `has_embedding(owner_type: str, owner_id: str, text_hash: str) -> bool` — used to skip re-embedding unchanged text.
  - `InMemoryEmbeddingStore` implementing `EmbeddingStore` using cosine similarity over stored float lists; this is the test backend for all unit tests.
  - `NullEmbeddingStore` — no-op implementation returned when `embeddings_enabled` is `False`; `search_similar` always returns `[]`.

Acceptance criteria for 12a:
- `InMemoryEmbeddingStore.search_similar` returns results ranked by cosine similarity.
- `NullEmbeddingStore.search_similar` always returns an empty list.
- Unit tests run without Postgres or Ollama.

### ✅ 12b — Embedding lifecycle

- Define where and when `provider.embed()` is called:
  - Embeddings are computed **only** when `runtime_config.embeddings_enabled` is `True`.
  - The executor calls `embed_and_persist(text, owner_type, owner_id, runtime_config, provider, embedding_store)` as a post-write step after two events: (1) a memory item is promoted via `write_memory_item(...)`, and (2) a `model` or `compress` node artifact is persisted.
  - Tool output artifacts are not embedded by default (they are structured data, not free text).
  - `embed_and_persist` checks `embedding_store.has_embedding(owner_type, owner_id, text_hash)` before calling `provider.embed()` to avoid re-embedding unchanged content.
  - Add `embed_and_persist(text, owner_type, owner_id, runtime_config, provider, embedding_store) -> EmbeddingRef | None` as a standalone function in `llmasm/storage/embeddings.py`. Returns `None` when `embeddings_enabled` is `False`.

Acceptance criteria for 12b:
- With `embeddings_enabled=False`, `embed_and_persist` returns `None` and never calls `provider.embed()`.
- With `embeddings_enabled=True` and a fake provider, `embed_and_persist` persists an `EmbeddingRef` and the vector.
- A second call with identical text and owner skips `provider.embed()`.

### ✅ 12c — Text-based memory retrieval (default path)

- Add memory retrieval functions in `llmasm/storage/memory.py` (and later mirrored in postgres.py):
  - `retrieve_workspace_context(workspace_graph_id, query, budget_tokens, filters) -> list[ContextItem]`;
  - `search_memory(workspace_graph_id, query, filters, limit) -> list[MemoryItem]`.
- Text-overlap implementation: score memory items by word-overlap ratio against the query string, apply graph filters (`active_goal_id`, `task_type`, `scope`), sort descending, trim to `budget_tokens`.
- This is the permanent baseline path, not a temporary stub. It runs when `embeddings_enabled` is `False` and is always present as the fallback when vector search returns no results.
- Wire `select_context` in `llmasm/runtime/context.py` to call `search_memory` for the text path and `embedding_store.search_similar` for the vector path. Both paths return `list[ScoredMatch]`; they are merged and re-ranked before token trimming. Update `select_context` signature: `select_context(run, node, direct_inputs, embedding_store: EmbeddingStore)`; callers always pass an `EmbeddingStore` — either `InMemoryEmbeddingStore` (real) or `NullEmbeddingStore` (disabled). The Phase 9 text-overlap stub is replaced by this call.

Acceptance criteria for 12c:
- Follow-up prompt retrieves prior summary and not unrelated artifacts.
- Context selector respects token budget.
- With `NullEmbeddingStore`, text-overlap ranking is used.
- With `InMemoryEmbeddingStore` containing embeddings, vector-ranked results appear before lower-scored text matches.

### ✅ 12d — Optional pgvector backend

> **Gap closed 2026-05-30:** `002_pgvector.sql` now contains only `CREATE EXTENSION`.
> Column DDL (`vector vector(N)`) is applied at `PostgresEmbeddingStore.__init__` time
> using `RuntimeConfig.embedding_dimensions` (default 768). A startup guard raises
> `StorageError` if the existing column has a different dimension count.

- Add `llmasm/storage/migrations/002_pgvector.sql`:
  - check `CREATE EXTENSION IF NOT EXISTS vector` (no-op if already present);
  - add `vector vector({dimensions})` column to `embeddings` (applied only when pgvector is available; migration is skipped if extension is absent);
  - add `ivfflat` or `hnsw` index on the vector column.
- Add `PostgresEmbeddingStore(EmbeddingStore)` in `llmasm/storage/postgres.py`:
  - `persist`: insert into `embeddings`; write to `vector` column when pgvector is available, else write float array to `vector_json`;
  - `search_similar`: use `ORDER BY vector <=> query_vector LIMIT n` when pgvector is available, else load all rows and compute cosine similarity in Python as a fallback.
- `run_migrations()` detects pgvector availability and skips `002_pgvector.sql` when the extension is absent.

Acceptance criteria for 12d:
- Integration tests tagged `@pytest.mark.pgvector` are skipped unless `PGVECTOR_AVAILABLE=1`.
- With pgvector available, `PostgresEmbeddingStore.search_similar` returns results ordered by vector distance.
- Without pgvector, the same test passes using the Python cosine fallback.

### ✅ 12e — Memory promotion

- Add `write_memory_item(workspace_graph_id, kind, text, source_artifact_id, source_run_id, confidence, runtime_config, provider, embedding_store) -> MemoryItem`:
  - persists `MemoryItem` via `Storage`;
  - calls `embed_and_persist` for the item text;
  - does **not** auto-promote all artifacts; callers must invoke this explicitly.

Acceptance criteria for 12e:
- `write_memory_item` persists both the `MemoryItem` and an `EmbeddingRef` when `embeddings_enabled=True`.
- `write_memory_item` skips embedding when `embeddings_enabled=False`.

## ✅ Phase 13: Run Analysis API

Goal: make execution inspection queryable and useful.

Implementation tasks:

- Add `llmasm/analysis/run.py`:
  - `query_run(run_id) -> RunAnalysis`;
  - include workspace, task graph, task edges, workspace edges, node states, artifacts, tool calls, model calls, checkpoints, token usage, failed nodes.
- Add convenience methods:
  - `failed_nodes()`;
  - `token_usage()`;
  - `context_used_by_model_call(model_call_id)`;
  - `expansions_for_run(run_id)`;
  - `follow_up_chain(task_graph_id)`.

Acceptance criteria:

- Querying a successful run returns all executed artifacts and checkpoints.
- Querying a failed run identifies the failed node state and error.
- Querying an expansion run identifies the source reasoning node and injected nodes.

## ✅ Phase 14: Public API And Examples

Goal: expose a small ergonomic API without hiding the graph model.

Implementation tasks:

- Define public facade:
  - `LLMASM(storage, provider_registry, tool_registry, runtime_config)`;
  - `compile(workspace_id, prompt) -> task_graph_id` — raises `CompilationError` if the repair loop exhausts all attempts;
  - `run(task_graph_id) -> run_id` — raises `ExecutionError` if a fatal node failure stops the run;
  - `ask(workspace_id, prompt) -> FinalAnswer` — combines compile and run; propagates `CompilationError` or `ExecutionError` transparently without swallowing them;
  - `query_run(run_id) -> RunAnalysis`.
- Provide examples:
  - `examples/conversation_summary.py`;
  - `examples/weather_followup.py`;
  - both examples use fake tools by default.
- Document how to plug in Ollama.

Acceptance criteria:

- A user can run the fake examples without Postgres or Ollama.
- An Ollama example is available but guarded by environment variables.
- Public API remains thin over compiler, storage, and executor components.

## ⏸ Phase 15: Optional Apache AGE Backend

> **Deferred.** The extension is installed in the Docker image and the `llmasm_graph`
> AGE graph is created by `initdb/001_extensions.sql`, but there is zero usage in Python
> code. Implement after V0 is validated end-to-end.

Goal: add graph-query acceleration without changing the core API.

Implementation tasks:

- Add storage adapter or query helper for Apache AGE.
- Keep plain Postgres as the compatibility baseline.
- Mirror `workspace_edges` and `task_edges` into AGE graph structures when configured.
- Implement traversal methods through AGE when available.
- Fall back to recursive SQL when AGE is unavailable.

Acceptance criteria:

- Existing tests pass with plain Postgres.
- AGE-specific tests are skipped unless extension is installed.
- Traversal returns equivalent results for SQL and AGE backends on the same fixture.

## Cross-Phase Test Matrix

Required unit test groups:

- Model validation.
- Schema registry and transforms.
- Tool registry.
- Goal classification.
- Proposal parsing.
- Compiler repair loop.
- Semantic validator error codes.
- Scheduler frontier selection.
- Context selection budget behavior.
- Context selection: text-overlap path with `NullEmbeddingStore`.
- Context selection: vector-ranked path with `InMemoryEmbeddingStore`.
- `EmbeddingStore` cosine similarity correctness.
- Embedding lifecycle: deduplication by text hash.
- Runtime execution state transitions.
- Runtime expansion validation.
- Storage contract tests.
- Run analysis queries.

Required integration scenarios:

- Compile and execute: `"retrieve the conversation xyz and give me a summary of the content"`.
- Follow-up: `"now check if the weather was actually raining"`.
- Missing conversation routes to final answer without summarizer.
- Planner emits invalid graph, repair loop fixes it.
- Planner emits mismatched `goal_action`, compilation fails.
- Expansion injects weather lookup.
- Expansion with unknown tool fails.
- Context budget overflow fails clearly.

## Implementation Order For Coding Agents

1. Implement phases strictly in order.
2. Do not implement Postgres before the in-memory storage contract is stable.
3. Do not implement runtime expansion before basic execution works.
4. Do not add Apache AGE before plain Postgres works.
5. Keep each phase independently testable.
6. When a phase introduces an interface, write fake implementations before external integrations.
7. Prefer small commits by phase or subphase.

## V0 Completion Definition

**Status: complete** — all checklist items below are satisfied.

V0 is complete when:

- A workspace graph can be created.
- A prompt can compile into a validated task subgraph through a mock or Ollama planner.
- A task graph can execute through the VM with fake tools and fake model provider.
- Tool/model calls, artifacts, checkpoints, run node states, task edges, and workspace edges are persisted.
- A follow-up prompt links to prior context and active goal.
- A reasoning node can inject a validated new tool node.
- Run analysis can explain what happened.
- The full test matrix passes against in-memory storage and the required Postgres subset.
