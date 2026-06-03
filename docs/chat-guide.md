# llmasm Chat Guide

`examples/chat.py` is an interactive REPL that exercises the full llmasm pipeline:
compile → execute → persist → embed.  
It is the primary tool for observing how the library behaves end-to-end.

---

## Prerequisites

- Ollama running on `http://localhost:11434`
- At minimum one model pulled, e.g. `ollama pull gemma4:e4b`
- For persistent storage: Postgres stack running (`make e2e-stack-up`)
- For embeddings: an embedding model pulled, e.g. `ollama pull nomic-embed-text:v1.5`

---

## Invocation

```bash
python examples/chat.py [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--ollama-url URL` | `http://localhost:11434` | Ollama base URL |
| `--runtime-model MODEL` | `llama3.1:8b` | Model used by executor nodes (model, compress, final) |
| `--planner-model MODEL` | `llama3.1:8b` | Model used by the compiler to plan task graphs |
| `--embedding-model MODEL` | `nomic-embed-text` | Ollama embedding model |
| `--embedding-dimensions N` | `768` | Vector dimensions — must match the model's output and the existing pgvector column |
| `--embeddings` | off | Enable vector embedding of turns and context retrieval via pgvector |
| `--db-url DSN` | *(in-memory)* | PostgreSQL DSN for persistent storage |
| `--workspace-name NAME` | `chat` | Named workspace; reused across restarts when `--db-url` is set |
| `--fresh` | off | Append a Unix timestamp to `--workspace-name` to start a clean isolated session |
| `--compiler-attempts N` | `3` | Max repair-loop iterations the compiler may take before giving up |
| `--context-turns N` | `10` | Max memory items fed as context per turn (0 = unlimited) |
| `--timeout SECS` | `180.0` | Per-request timeout passed to Ollama |

### Typical invocations

**Minimal (in-memory, no embeddings):**
```bash
python examples/chat.py --runtime-model gemma4:e4b --planner-model gemma4:e4b
```

**Persistent session (resume last workspace named `chat`):**
```bash
python examples/chat.py \
  --db-url postgresql://llmasm:llmasm@localhost:15432/llmasm \
  --runtime-model gemma4:e4b --planner-model gemma4:e4b
```

**Fresh session with embeddings:**
```bash
python examples/chat.py \
  --db-url postgresql://llmasm:llmasm@localhost:15432/llmasm \
  --runtime-model gemma4:e4b --planner-model gemma4:e4b \
  --embeddings --embedding-model nomic-embed-text:v1.5 --embedding-dimensions 768 \
  --fresh
```

---

## REPL Commands

| Command | Effect |
|---------|--------|
| `/help` | List available commands |
| `/quit` | Exit (also Ctrl-D) |
| `/clear` | Close the active goal and reset `FOLLOWS_UP` chain |
| `/inspect` | Print all memory items and workspace edges in the current workspace |
| `/inject <text>` | Immediately persist a `human_note` memory item before the next turn |

Any other input is compiled into a task graph and executed.

---

## Available Tools

Chat auto-registers four tools at startup. The planner sees them in its prompt and can generate `tool` nodes.

| Tool name | Input schema | Output schema | Data source |
|-----------|-------------|---------------|-------------|
| `wikipedia.search` | `RawText` | `RawText` | [Wikipedia MediaWiki API](https://en.wikipedia.org/w/api.php) (free, no key) |
| `weather.lookup` | `WeatherQuery` | `WeatherObservation` | [Open-Meteo](https://open-meteo.com) (free, no key) |
| `calculator.eval` | `RawText` | `RawText` | Python `eval` with restricted `math` namespace |
| `file.read` | `RawText` | `RawText` | Local filesystem, UTF-8, 1 MB limit |

Each tool returns a result object or an error message — the executor marks the node as `succeeded` either way. Tool caching applies to `calculator.eval` and `file.read` by default (same inputs → cached artifact).

---

## What happens on each turn

1. **Compile** — the planner model generates a task graph JSON from the prompt and any few-shot examples retrieved from the workspace.  If the graph is invalid, the repair loop runs up to `--compiler-attempts` times.
2. **Execute** — the scheduler runs each node in topological order.  Context for model/compress nodes is assembled by `select_context`, which merges vector search results (when `--embeddings`) with word-overlap search and trims to the token budget.
3. **Persist** — after execution, the Q&A pair is stored as a `MemoryItem` via `write_memory_item`.  When `--embeddings` is set, the text is also embedded and stored in pgvector so future turns can retrieve it semantically.
4. **Edge wiring** — a `FOLLOWS_UP` edge links the new task graph to the previous one; a `PRODUCED` edge links the final node to the new memory item.

The footer of each turn shows: `turn N  |  goal: <action>  |  tg: <short id>`.

---

## Test Conversations

Each conversation below is designed to probe a specific aspect of the library.
Run each with `--fresh` to start from an empty workspace.

---

### 1. Baseline: single-turn answer

**Purpose:** verify the compiler can produce a minimal task graph and the executor can produce a non-empty answer.

```
[0]> What is the speed of light in a vacuum?
```

**What to observe:**
- The compiler emits a graph with an `intent`, `model`, and `final` node.
- The executor runs all three nodes.
- The answer contains "299,792,458 m/s" or equivalent.
- `goal:` in the footer shows `new` (first turn, no prior goal).

---

### 2. Goal persistence across turns

**Purpose:** verify the goal tracker carries context from turn to turn and the `FOLLOWS_UP` edge is wired.

```
[0]> I want to understand how neural networks learn.

[1]> Can you explain backpropagation in more detail?

[2]> What is the role of the learning rate?

[3]> /inspect
```

**What to observe:**
- After turn 0, `goal:` changes from `new` to a descriptive label (e.g. `continue` or `deepen`).
- `/inspect` shows at least three `MemoryItem` rows (one per turn) and two `FOLLOWS_UP` workspace edges.
- Turn 1 and 2 answers reference prior context without the user re-stating it.

---

### 3. Context retrieval — text-overlap path (no embeddings)

**Purpose:** verify that `retrieve_workspace_context` surfaces relevant prior turns through word-overlap scoring alone.

Run **without** `--embeddings`.

```
[0]> The project codename is Nighthawk and it uses a graph-based VM to execute LLM task graphs.

[1]> What is 2 + 2?

[2]> Tell me about the project architecture again.
```

**What to observe:**
- Turn 2 answer references "Nighthawk" even though the prompt does not, because the turn 0 memory item scores high on word overlap against the query.
- Turn 1 answer does not mention Nighthawk (low relevance, different topic).

---

### 4. Context retrieval — vector path (with embeddings)

**Purpose:** verify that `search_similar` surfaces semantically related turns even when surface words differ.

Run **with** `--embeddings`.

```
[0]> The project codename is Nighthawk and it uses a graph-based VM to execute LLM task graphs.

[1]> Remind me about the execution engine for the system.
```

**What to observe:**
- The answer to turn 1 references "Nighthawk" and "graph-based VM" despite the query using different words ("execution engine", "system").
- Run `psql` to confirm a row exists in `embeddings` for the turn 0 memory item:
  ```sql
  SELECT owner_type, dimensions, text_hash FROM embeddings ORDER BY created_at DESC LIMIT 3;
  ```

---

### 5. Cross-workspace isolation

**Purpose:** verify that `--fresh` prevents memory bleed from previous sessions.

Run the following in two separate terminals:

**Session A:**
```bash
python examples/chat.py --db-url ... --embeddings ... --workspace-name isolation-test
```
```
[0]> The secret ingredient is saffron.
```

**Session B (--fresh):**
```bash
python examples/chat.py --db-url ... --embeddings ... --workspace-name isolation-test --fresh
```
```
[0]> What is the secret ingredient?
```

**What to observe:**
- Session B answer does not mention "saffron" — it has no access to Session A's memory items because the workspace ID is different.

---

### 6. Compiler repair loop

**Purpose:** trigger the compiler's repair cycle by providing a deliberately ambiguous prompt that may produce an invalid graph on the first attempt.

```
[0]> Retrieve the summary of document XYZ and then check if the author name matches the one in the external registry and produce a final answer combining both.
```

**What to observe:**
- The compiler may emit one or more repair attempts (look for debug prompt files in `data/*/debug_prompts/` if debug logging is enabled).
- A valid graph is eventually compiled and executed, or `CompilationError` is raised if all attempts fail.
- The footer shows the task graph short ID; inspect the graph via `/inspect` or `psql`.

---

### 7. `/inject` — manual context seeding

**Purpose:** verify that a manually injected context note is visible to the next turn without going through the compiler.

```
[0]> /inject The user's preferred language for examples is TypeScript.

[1]> Show me a hello world example.
```

**What to observe:**
- The code example in the answer uses TypeScript, not Python or any other language.
- `/inspect` shows a `human_note` memory item with the injected text.

---

### 8. Dimension mismatch guard

**Purpose:** confirm the startup guard prevents connecting to a workspace whose pgvector column was initialised with a different model's dimensions.

First, start a session with 768-dim embeddings and send at least one turn (to create the column):
```bash
python examples/chat.py --db-url ... --embeddings --embedding-dimensions 768 --workspace-name dim-test
```

Then attempt to connect with a different dimension:
```bash
python examples/chat.py --db-url ... --embeddings --embedding-dimensions 384 --workspace-name dim-test
```

**What to observe:**
- The second invocation raises `StorageError: embeddings.vector column has 768 dimensions but RuntimeConfig.embedding_dimensions=384` and exits immediately, before any prompt is processed.

---

### 9. Tool-driven turns (Wikipedia + Calculator)

**Purpose:** verify the planner generates tool nodes when tools are registered and the executor invokes them.

Run **without** `--embeddings` (tools work regardless of embedding state).

```
[0]> When was Alan Turing born?

[1]> What is the factorial of 10?

[2]> /graph
```

**What to observe:**
- Turn 0 response includes Turing's birth year (1912) from Wikipedia.
- Turn 1 response shows `3628800`.
- `/graph` after turn 2 shows at least one node with `kind=tool` and `status=succeeded`. The tool node may be named `wikipedia.search` or `calculator.eval`.
- If the planner omits a tool node and answers from the model's own knowledge, the answer should still be correct — the compiler prompt says "If NO tools are listed, do not generate tool nodes", so with tools present the planner should prefer them for retrieval/computation.

*Repeat with the weather tool:*
```
[3]> What's the current weather in London?
```

**What to observe:**
- If Open-Meteo is reachable, the response includes a temperature and condition (e.g., "15°C, partly cloudy").
- The footer shows `goal: new` or `goal: continue`.

---

## Inspecting state via psql

```bash
psql postgresql://llmasm:llmasm@localhost:15432/llmasm
```

Useful queries:

```sql
-- All workspaces
SELECT id, name, status, created_at FROM workspace_graphs ORDER BY created_at DESC;

-- Memory items for a workspace
SELECT id, kind, left(text, 80) AS preview, created_at
FROM memory_items WHERE workspace_graph_id = '<id>'
ORDER BY created_at;

-- Embeddings written in the last session
SELECT owner_type, owner_id, model, dimensions, created_at
FROM embeddings ORDER BY created_at DESC LIMIT 10;

-- Workspace edges (conversation links)
SELECT edge_type, from_type, left(from_id,20) AS from_id,
       to_type, left(to_id,20) AS to_id
FROM workspace_edges WHERE workspace_graph_id = '<id>';

-- Run summary
SELECT r.id, r.status, count(ns.node_id) AS nodes,
       count(a.id) AS artifacts
FROM runs r
LEFT JOIN run_node_states ns ON ns.run_id = r.id
LEFT JOIN artifacts a ON a.run_id = r.id
GROUP BY r.id ORDER BY r.id DESC LIMIT 5;
```
