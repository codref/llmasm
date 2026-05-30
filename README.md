# LLMASM

**LLM Assembly Scheduling Machine** — local LLM orchestration with persistent graph memory.

LLMASM is a Python library for orchestrating local LLM and tool calls inside a
persistent typed graph. A user prompt creates or extends a bounded task subgraph
inside a long-lived workspace graph that grows across prompts, follow-ups, tool
calls, model calls, artifacts, memories, and reasoning steps.

The central design goal is **context discipline for local LLMs**: instead of
passing the full conversation or history to every model call, the runtime
selects only the required upstream artifacts, goal-relevant memory, and local
neighborhood of the graph.

## Features

- **Hybrid prompt-to-subgraph compiler** — compiles natural-language tasks into
  typed executable subgraphs with deterministic validation and an automatic
  repair loop
- **VM-style executor** — program counter, checkpoints, and deterministic step
  progression through the task graph
- **Persistent execution traces** — every plan, node execution, tool call, model
  call, artifact, and checkpoint is persisted in Postgres
- **Ollama-first local LLM support** — structured output enforcement via JSON
  Schema for planner calls
- **Context selection** — combines graph traversal and embedding-assisted
  retrieval to keep model prompts small and relevant
- **Runtime graph expansion** — ReAct-style reasoning nodes can propose and
  inject new nodes during execution
- **Follow-up linking** — follow-up prompts connect to prior task graphs, active
  goals, and memory items instead of creating isolated runs
- **Run analysis** — queryable execution traces for debugging and inspection

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) (for model inference)
- Postgres (storage backend; pgvector and Apache AGE are optional enhancements)

## Installation

```bash
pip install llmasm
```

Or from source:

```bash
git clone <repo-url> llmasm
cd llmasm
pip install -e ".[dev]"
```

## Quick Start

```python
from llmasm import LLMASM
from llmasm.storage.memory import InMemoryStorage
from llmasm.config import RuntimeConfig

# Create a workspace and ask a question
llmasm = LLMASM(storage=InMemoryStorage())
answer = llmasm.ask("retrieve the conversation xyz and give me a summary")
print(answer.text)
```

See `examples/conversation_summary.py` for a complete runnable example using
fake tools — no Postgres or Ollama required.

## Examples

| Example | Description |
|---------|-------------|
| `conversation_summary.py` | Minimal example with fake tools and fake model |
| `complex_end_to_end.py` | Full pipeline with deterministic or Ollama planner, real tool execution, and optional graph viewer export |
| `open_dataset_qa.py` | Autonomous SQuAD 2.0 QA benchmark with local Wikipedia retriever |
| `hotpot_multihop_qa.py` | Multi-hop HotpotQA benchmark with conversation-demo mode and workspace memory |

### Interactive chat with Postgres and embeddings

Run the chat interface backed by Postgres with vector-embedding context retrieval
(requires Ollama with the embedding model pulled):

```bash
# Default: nomic-embed-text, 768 dimensions
python examples/chat.py \
  --db-url postgresql://llmasm:llmasm@localhost:15432/llmasm \
  --embeddings

# Custom model and dimensions
python examples/chat.py \
  --db-url postgresql://llmasm:llmasm@localhost:15432/llmasm \
  --embeddings \
  --embedding-model mxbai-embed-large \
  --embedding-dimensions 1024
```

> **Note:** `--embedding-dimensions` must match the model's actual output size.
> Changing it after a workspace has been created raises a `StorageError` — drop
> the `vector` column and re-initialise to switch models.

## Architecture

```
Prompt → Compiler → TaskGraph → Runtime VM → Final Answer
                  ↓                            ↓
            WorkspaceGraph ←── Persistence ───┘
            (long-lived, growing across prompts)
```

- **Compiler** — classifies the prompt goal, retrieves relevant prior context,
  assembles a planner prompt, calls the planner model with structured output
  enforcement, validates the proposed task subgraph, and persists it before
  execution
- **Runtime VM** — executes nodes when their inputs are available, persists
  artifacts and checkpoints after each step, handles retryable and fatal errors,
  and validates runtime expansion requests from reasoning nodes
- **Context Selection** — ranks inputs by priority (required edges, active goal,
  node instructions, relevant memories, graph neighborhood), trims to the
  model's token budget, and never passes full conversation history
- **Storage** — plain Postgres fallback schema with optional pgvector and Apache
  AGE backends; artifacts are append-only, run state is mutable

## Development

```bash
# Run tests
make test

# Lint
make lint

# Type check
make typecheck

# Start the local e2e stack (Postgres + graph viewer)
make e2e-stack-up

# Stop the local e2e stack
make e2e-stack-down
```

### Postgres integration tests

The e2e stack exposes Postgres on port 15432. Once it's up, run the integration
test suite with:

```bash
LLMASM_TEST_DB=postgresql://llmasm:llmasm@localhost:15432/llmasm \
    python -m pytest tests/unit/test_postgres_storage.py -v
```

Tests are skipped automatically when `LLMASM_TEST_DB` is not set.

## Documentation

- [RFC](docs/llmasm-rfc.md) — full specification: core concepts, compiler,
  runtime VM, storage design, context selection, failure modes
- [Implementation Plan](docs/llmasm-implementation-plan.md) — phased build plan
  with module boundaries and acceptance criteria
- [End-to-End Testing](docs/e2e.md) — running the complex e2e example, graph
  viewer, Postgres stack, and autonomous QA benchmarks

## License

[License details here]
