# Chat interface plan

## Goal

A minimal chat REPL where each turn feeds only relevant context to each node — no pre-composed
conversation history.  The workspace graph grows across turns through MemoryItems and
workspace edges, and the compiler / runtime select context via the existing retrieval
pipeline.

## How the compiler already does what we need

`llmasm/compiler/compiler.py:65-94` per turn:

| Step | Code | What it does |
|------|------|-------------|
| Classify goal | `classify_goal_action(prompt, active_goal)` | NEW / CONTINUE / STEER |
| Prior context | `_prior_context(workspace_id, prompt)` | retrieves MemoryItems |
| Planner prompt | `render_planner_prompt(...)` | planner sees goal + context |
| Planner decides | `compile_with_repair(...)` | simple `intent→model→final` vs tool-loop vs expand DAG |

The planner **already decides** "simple reply" vs "loop a tool".  No pre-classification needed.

## Gap: runtime model prompt drops workspace context

`llmasm/runtime/executor.py:185-194`:

```python
selected = select_context(...)                         # MemoryItems fetched here
prompt = self._render_node_prompt(node, selected.direct_inputs)  # selected.items dropped
```

`_render_node_prompt` builds `{"instruction": ..., "inputs": ...}` but never serializes the
workspace context items.  The fix is to pass `selected.items` through and include it in the
prompt JSON.

## Changes

### A. `llmasm/runtime/executor.py` — include workspace context in model prompts

| What | Where |
|------|-------|
| Add `context_items` parameter to `_render_node_prompt()` | signature around line 324 |
| Serialize context text into the prompt dict | method body |
| Pass `selected.items` at the call site | line 194 |

Before:
```python
prompt = self._render_node_prompt(node, selected.direct_inputs)
```

After:
```python
prompt = self._render_node_prompt(node, selected.direct_inputs, selected.items)
```

### B. `examples/chat.py` — thin loop, graph-edge persistence

| Change | Why |
|--------|-----|
| Remove `build_conversation_history()` / `build_prompt()` | no pre-composed history |
| Pass **raw user prompt** to the pipeline | let graph handle context |
| Switch `ask()` → `compile()` + `run()` + `query_run()` | need `task_graph_id`/`run_id` for edges |
| Create `FOLLOWS_UP` workspace edges between consecutive task graphs | structural cross-turn links |
| Create `PRODUCED` workspace edges from final nodes to MemoryItems | provenance |
| Keep MemoryItem persistence | already correct |
| Keep `--context-turns` flag | controls retrieval budget, not a pre-composed string |
| `/clear` closes active goal, starts a new one | goal lifecycle |

Same pattern as `examples/hotpot_multihop_qa.py:380-444` but without custom tools.

## Per-turn flow after changes

```
user input
  │
  ▼
classify_goal_action()         ← deterministic: NEW / CONTINUE / STEER
  │
  ▼
_prior_context()               ← retrieves MemoryItems from workspace
  │
  ▼
planner prompt (goal + context + prompt)
  │
  ▼
planner LLM decides DAG shape
  │  ├─ simple:    intent → model → final
  │  ├─ tool-loop: intent → tool → model → final
  │  └─ expansion: intent → expand { tool → model → observation → ... } → final
  │
  ▼
runtime execute
  │  select_context()          ← fetches MemoryItems again
  │  _render_node_prompt()     ← includes context text in model prompt
  │  model.generate()          ← sees relevant context, not full history
  │
  ▼
FinalAnswer
  │
  ▼
persist MemoryItem             ← Q + A stored for future retrieval
persist workspace edges        ← follows_up / produced links
```

## Context bounding

- `RuntimeConfig.default_context_tokens` (4096) limits how much workspace context the runtime
  selects per model call.
- `_prior_context()` has a hard-coded 800-token budget for the planner prompt.
- `--context-turns` in the chat controls how many MemoryItems retention is allowed (0 =
  unlimited).
- MemoryItems are always persisted to the graph; bounding only affects what the retrieval
  functions select.

## Dependencies

No new dependencies.  The chat already imports `prompt-toolkit>=3` and `rich>=13`.
