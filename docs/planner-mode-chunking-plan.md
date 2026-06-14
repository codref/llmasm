# Plan: Chunking and Summary Nodes for Planner Mode (`LLMASM.ask`)

**Status:** design / ready for implementation  
**Scope:** library-level `LLMASM.ask`, not only the chat example  
**Decisions already taken:**

1. Chunking happens inside `LLMASM.ask` when a user prompt is classified as a long source document.
2. After chunking, the **planner receives a placeholder**, not the original full text. This keeps planner prompts small for models like `gemma4:e4b-it-qat`.
3. The **summary is produced by a planner-emitted node**, not by a pre-ingestion fast-path graph.
4. The chat example must **stop storing the full `Q: ...\nA: ...` as a single `turn` memory item**.
5. This plan is intentionally pragmatic. A future session should evaluate whether a pre-ingestion summary graph or a hybrid approach works better.

---

## Problem Statement

`LLMASM.ask()` currently feeds the raw user prompt to the planner and stores each turn as a monolithic `kind="turn"` memory item. For long documents this causes two independent blow-ups:

- **Planner prompt bloat:** the full transcript is in the `User prompt` section of every planner call.
- **Runtime context bloat:** the turn memory item contains the entire transcript, so `select_context` / `retrieve_workspace_context` can inject it into model prompts on every subsequent question.

The fast path already solves this by chunking long `source_passage` items and retrieving only relevant chunks. Planner mode needs the same treatment.

---

## Non-Goals

- Do not remove or break the fast-path chunking implementation.
- Do not change the `Storage` Protocol or Postgres migrations.
- Do not redesign `EmbeddingStore`; reuse `InMemoryEmbeddingStore` / `PostgresEmbeddingStore` as-is.
- Do not require embeddings to be enabled for chunking to work.
- Do not implement map-reduce summarization for very large documents in this phase (single summary node only).

---

## Target Behavior

For `LLMASM.ask(workspace_id, prompt)`:

1. If the prompt is a long source document, chunk it and store the chunks as `source_passage` memory items.
2. Replace the prompt seen by the planner with a short placeholder so the planner prompt stays small.
3. Allow the planner to emit a `model` node whose job is to summarize the source; store that summary as a `source_passage` with `is_summary=True`.
4. On follow-up questions, runtime context selection surfaces the summary first, then the most relevant chunks.
5. Do not store the full source text inside a `turn` memory item.

---

## Phase 1: Source-Document Ingestion in `LLMASM.ask`

### Where to change

- `llmasm/api.py` — `LLMASM.ask()`
- New helper module: `llmasm/conversation/ingestion.py`

### Implementation

Create `llmasm/conversation/ingestion.py` with a helper such as:

```python
def maybe_ingest_long_source(
    workspace_graph_id: str,
    prompt: str,
    *,
    storage: Storage,
    provider: LLMProvider,
    runtime_config: RuntimeConfig,
    embedding_store: EmbeddingStore | None,
    turn: int | None = None,
) -> str:
    """Chunk and store a long source prompt; return a placeholder for the planner."""
```

Behavior:

1. Classify the prompt as source using `llmasm.conversation.classifier.classify_dialogue` (or a token-count heuristic if classification is disabled).
2. Count tokens with `runtime_config.tokenizer`.
3. If `chunking_enabled` and tokens > `chunking_trigger_tokens`:
   - Generate `source_id = new_id("memory")`.
   - Chunk with `SentenceTextChunker` using `chunk_target_tokens` / `chunk_overlap_tokens`.
   - Persist chunks via `store_source_passages(...)`.
   - Return a placeholder like:
     ```
     [A long source document has been stored in workspace chunks. Ask questions about it.]
     ```
4. Otherwise return the original `prompt` unchanged.

### `LLMASM.ask` change

```python
def ask(self, workspace_id: str, prompt: str) -> FinalAnswer:
    effective_prompt = maybe_ingest_long_source(
        workspace_id,
        prompt,
        storage=self.storage,
        provider=self.provider,
        runtime_config=self.runtime_config,
        embedding_store=self.embedding_store,
    )
    task_graph_id = self.compile(workspace_id, effective_prompt)
    run_id = self.run(task_graph_id)
    ...
```

**Acceptance:**

- A prompt above the trigger is split into multiple `source_passage` memory items.
- The planner prompt artifact for the same turn does not contain the full original text.
- A short prompt is unchanged.

---

## Phase 2: Planner-Emitted Summary Node

### Where to change

- `llmasm/compiler/prompt.py` — planner prompt rules
- `llmasm/compiler/compiler.py` — `_canonical_node_fields` / validation (if needed)
- `llmasm/conversation/memory.py` — helper to store summary after execution

### Implementation

Add a rule in `render_planner_prompt`:

```
- When the user has supplied a long source document (indicated by the placeholder "[A long source document has been stored...]"), include a `model` node named "summarize_source" early in the graph. It should read the source placeholder and output schema "Summary". Its instruction should ask for a concise summary of the stored source. Connect it to downstream QA model nodes as needed.
```

The planner already knows how to emit `model` nodes; we only need to:

1. Allow the planner to recognize the placeholder.
2. Ensure a `Summary` output schema is acceptable for a model node.
3. After execution, detect any artifact produced by a node whose name or metadata indicates it is a summary (e.g. `metadata.is_summary_node = True`), validate it as `Summary`, and persist it via `store_source_summary(...)` linked to the same `source_id` used during ingestion.

Suggested detection hook: in `LLMASM.ask`, after `self.run(task_graph_id)`:

```python
for artifact in self.storage.list_artifacts(run_id):
    if artifact.metadata.get("is_summary_node"):
        summary_text = Summary.model_validate(artifact.content_json).text
        store_source_summary(
            workspace_graph_id=workspace_id,
            summary_text=summary_text,
            source_id=artifact.metadata["source_id"],
            ...
        )
```

The compiler should canonicalize `metadata.is_summary_node` and `metadata.source_id` from the planner proposal into the persisted `Node.metadata`.

**Acceptance:**

- When the placeholder is present, the planner emits a graph containing a model node whose output schema is `Summary`.
- After execution, a `source_passage` item with `is_summary=True` exists and is linked to the chunks via `source_id`.

---

## Phase 3: Runtime Context Prefers Summary + Chunks

### Where to change

- `llmasm/runtime/context.py` — `select_context`

### Implementation

`select_context` already retrieves `MemoryItem`s via vector search and word overlap. Update it to:

1. Separate retrieved `source_passage` items into:
   - `summary_items` where `metadata.get("is_summary") is True`
   - `chunk_items` where `metadata.get("is_summary") is not True`
2. If a summary exists, reserve a small fixed budget for it (e.g. 128 tokens) and place it first in the context.
3. Fill the remaining budget with the highest-scoring chunks.
4. Continue to mix in `user_question` / `assistant_answer` and artifact context as before.

Keep the existing `default_context_tokens` budget behavior; do not increase prompt size.

**Acceptance:**

- A follow-up question sees the summary in its context.
- Relevant chunks are still retrieved via embeddings/word overlap.
- The full original transcript is never injected as one item.

---

## Phase 4: Stop Storing Monolithic `turn` Memory Items

### Where to change

- `examples/chat.py` — non-fast-path loop (file-driven and REPL)

### Implementation

Replace:

```python
memory = write_memory_item(
    workspace_graph_id=workspace_id,
    kind="turn",
    text=f"Q: {question}\nA: {answer.text}",
    ...
)
```

with:

```python
store_user_question(
    workspace_graph_id=workspace_id,
    text=question,
    storage=storage,
    runtime_config=app.runtime_config,
    provider=app.provider,
    embedding_store=app.embedding_store,
    source_run_id=run_id,
    turn=turn_idx,
)
store_assistant_answer(
    workspace_graph_id=workspace_id,
    text=answer.text,
    storage=storage,
    runtime_config=app.runtime_config,
    provider=app.provider,
    embedding_store=app.embedding_store,
    source_run_id=run_id,
    turn=turn_idx,
)
```

If the question was a long source document that was chunked, do **not** store the original text as a `user_question`; the chunks already exist as `source_passage` items.

**Acceptance:**

- After a planner-mode conversation, workspace memory contains separate `user_question` / `assistant_answer` / `source_passage` / summary items.
- No `kind="turn"` item contains a multi-thousand-token source text.

---

## Phase 5: Tests

Add or extend tests in `tests/unit/`:

1. `test_ask_chunks_long_source` — `LLMASM.ask` with a long source prompt produces chunks and a placeholder planner prompt.
2. `test_planner_emits_summary_node` — a FakeProvider planner output containing a summary node is accepted and normalized.
3. `test_summary_artifact_stored_after_run` — after executing a graph with a summary node, a summary `MemoryItem` exists.
4. `test_select_context_prefers_summary` — given a summary and chunks, `select_context` includes the summary first.
5. `test_no_monolithic_turn_memory` — the chat example non-fast path stores separate memory items.

---

## Open Questions / Future Evaluation

- **Placeholder approach:** Passing a placeholder to the planner means the planner cannot reason about the document's actual content. If we later discover this hurts graph quality, evaluate replacing the placeholder with a short pre-computed summary or a hybrid where the first sentence of each chunk is shown.
- **Pre-ingestion summary graph:** The fast path builds a deterministic summary graph before the question. Consider whether planner mode should do the same when the user explicitly says "here is a document to ask about", so that the summary is available immediately rather than after the first planner-emitted graph.
- **Embedding requirement:** Chunking works without embeddings, but retrieval quality depends on them. Evaluate whether planner-mode chunking should force `embeddings_enabled=True` with a clear warning when it is disabled.
- **Token budget defaults:** The default `chunk_target_tokens=256` may still be too large for `gemma4:e4b-it-qat` on long contexts. Measure and possibly add a "small model" preset.
