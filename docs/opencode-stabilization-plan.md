# LLMASM Conversation Stabilization Plan

Purpose: implementation brief for opencode. The goal is to make LLMASM reliable for ordinary multi-turn chat and passage QA before adding back optional complexity such as embeddings, broad memory retrieval, and planner-driven graph synthesis for every turn.

## Problem Statement

The current library disappoints in planning, classification, and steering because too many responsibilities are delegated to the LLM planner and loose memory retrieval:

- The planner can fail before execution by emitting invalid graph IR: invented tools, invented schemas, missing ports, malformed routers, or non-JSON.
- Short conversational follow-ups such as `in what city?` and `and state?` are not reliably classified as continuations.
- Runtime context mixes source text, user turns, and prior assistant answers, so the model can answer from stale or bad assistant output.
- Embeddings add another retrieval signal without a conversation-state model, which can amplify irrelevant context.
- Grounded passage QA currently permits fallback to general knowledge in runtime prompts.

The immediate fix is not a better planner prompt. The immediate fix is a deterministic conversation path with explicit state and strict context rules.

## Non-Goals

- Do not remove the graph compiler, runtime, tool execution, routers, or planner repair loop.
- Do not redesign Postgres storage or migrations unless a storage protocol change requires it.
- Do not make embeddings work better in this phase. Disable them by default for the chat path.
- Do not implement a full agent framework or autonomous task decomposition.

## Target Behavior

For plain chat and passage QA:

1. The system must not ask the planner LLM to synthesize a new graph shape.
2. The graph shape should be deterministic: `intent -> model -> final`.
3. The system should maintain structured conversation memory:
   - `source_passage`: authoritative user-provided source text;
   - `user_question`: user questions;
   - `assistant_answer`: assistant answers;
   - `system_note` or `human_note`: optional injected context.
4. Passage QA answers should use `source_passage` as evidence and should not fall back to general knowledge unless explicitly configured.
5. Elliptical follow-ups should resolve against recent questions and source passages.
6. Planner-driven graphs should remain available for tool workflows and explicit orchestration tasks.

## Proposed Public Surface

Add a conversation-oriented API without breaking `LLMASM.ask()`:

```python
answer = app.chat(workspace_id, prompt)
```

Suggested optional config:

```python
RuntimeConfig(
    conversation_fast_path=True,
    grounded_qa_strict=True,
    chat_embeddings_enabled=False,
)
```

If adding config fields feels too broad, implement the same behavior inside `examples/chat.py` first, then promote it into the library after tests pass.

## Phase 1: Disable Embeddings For Chat

Implementation:

- In `examples/chat.py`, default the chat flow to no embeddings.
- Keep existing embedding flags for explicit opt-in, but make the displayed status clear.
- Ensure `RuntimeConfig.embeddings_enabled` remains false unless the user opts in.

Acceptance:

- Running `examples/chat.py` without flags does not call `provider.embed`.
- Existing embedding unit tests still pass.

## Phase 2: Add Structured Conversation Memory

Implementation:

- Add helpers for writing memory items with explicit kinds:
  - `source_passage`
  - `user_question`
  - `assistant_answer`
- Keep existing `turn` memory for backwards compatibility if needed, but do not use it as source evidence in the new chat path.
- Store metadata such as:
  - `turn`
  - `role`
  - `source_run_id`
  - `goal_action`
  - `is_authoritative_source`

Suggested location:

- `llmasm/conversation/memory.py`, or keep it local to `examples/chat.py` for the first pass.

Acceptance:

- A passage turn stores the passage as `source_passage`.
- A question turn stores the prompt as `user_question`.
- The answer is stored as `assistant_answer`.
- Prior assistant answers are not selected as evidence for grounded QA by default.

## Phase 3: Deterministic Dialogue Classifier

Implementation:

Create a small classifier for conversation mode. It should classify the prompt as:

- `source`: likely passage/context being supplied;
- `question`: direct question;
- `followup_question`: short elliptical question;
- `instruction`: task setup or steering instruction;
- `other`: fallback.

Rules should be deterministic first:

- A prompt ending in `?` is a question.
- Very short prompts with anaphora or conjunctions, such as `and state?`, `what city?`, `and why?`, are follow-up questions when there is recent question context.
- Long declarative text after a setup instruction is likely `source`.
- Prompts containing reading-comprehension setup language are `instruction`.

Suggested location:

- `llmasm/conversation/classifier.py`

Acceptance:

- The Staten Island session classifies:
  - setup prompt as `instruction`;
  - two long Staten Island paragraphs as `source`;
  - `How many burroughs are there?` as `question`;
  - `in what city?` as `followup_question`;
  - `and state?` as `followup_question`.

## Phase 4: Add Deterministic Chat Fast Path

Implementation:

- Add a method that bypasses planner graph synthesis for conversation mode.
- It should create a known valid graph equivalent to:

```text
intent -> model -> final
```

- The model node instruction should be generated by code based on dialogue state.
- For grounded QA, include only:
  - current question;
  - authoritative `source_passage` memory;
  - recent user questions for ellipsis resolution.

Do not include prior assistant answers as evidence.

Suggested location:

- `llmasm/conversation/chat.py`
- `LLMASM.chat()` in `llmasm/api.py`

Acceptance:

- The fast path cannot produce `UNKNOWN_TOOL`, `UNKNOWN_SCHEMA`, router errors, or missing-port compile errors.
- Existing `LLMASM.compile()` and `LLMASM.ask()` behavior remains unchanged.

## Phase 5: Strict Grounded Prompting

Implementation:

- Replace runtime wording for grounded QA. Avoid:

```text
If the context does not contain the answer, you may use your own knowledge.
```

- Use strict wording for the chat fast path:

```text
Answer using only the source passages below. If the answer is not present, say that the provided passage does not contain the answer.
```

- Keep non-strict behavior available for non-grounded chat.

Acceptance:

- Passage QA answers do not say `Based on general knowledge`.
- If source passages are empty, the answer asks for or reports missing source context.

## Phase 6: Regression Fixture From `session*.jsonl`

Implementation:

- Add a regression test using the Staten Island session.
- Use `FakeProvider` or a deterministic local fake that returns answer text based on prompt contents.
- Test classification, memory writes, and final answers separately.

Suggested tests:

- `tests/unit/test_conversation_classifier.py`
- `tests/unit/test_conversation_fast_path.py`

Acceptance:

- Expected answers:
  - `How many burroughs are there?` -> contains `five`;
  - `in what city?` -> contains `New York City`;
  - `and state?` -> contains `New York`.
- No planner calls are made for the fast-path test.
- No embedding calls are made unless explicitly enabled.

## Phase 7: Keep Planner For Orchestration

Implementation:

- Leave existing planner path untouched for now.
- Add a decision boundary:
  - use chat fast path for ordinary conversation and grounded QA;
  - use planner path when tools are registered and the user request clearly needs tool orchestration, branching, expansion, or multi-step graph execution.

Initial boundary can be conservative:

- `examples/chat.py` always uses fast path unless `--planner-chat` is passed.
- Library users can still call `compile()` directly.

Acceptance:

- Existing planner/compiler tests continue to pass.
- Chat examples become reliable for basic multi-turn QA.

## Verification Commands

Run after each phase:

```bash
make test
make lint
make typecheck
```

For focused iteration:

```bash
python -m pytest tests/unit/test_conversation_classifier.py -q
python -m pytest tests/unit/test_conversation_fast_path.py -q
```

Postgres tests are not required for this plan unless storage protocol changes are introduced.

## Suggested Implementation Order For opencode

1. Add conversation classifier and tests.
2. Add structured memory helpers and tests.
3. Add deterministic fast-path graph creation.
4. Add `LLMASM.chat()` or wire the fast path into `examples/chat.py`.
5. Make grounded QA prompting strict.
6. Default chat embeddings off.
7. Add the Staten Island regression test.
8. Run full verification.

## Success Criteria

The implementation is successful when the session represented by `session.jsonl` behaves as a coherent reading-comprehension conversation:

- setup prompt is acknowledged;
- passage chunks are retained as source text;
- questions are answered from the passage;
- short follow-ups remain attached to the same source context;
- no planner compile failures occur in the chat path;
- embeddings are not required for correctness.
