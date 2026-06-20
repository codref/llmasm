# Plan: Optimize Context Use for Local-First Models in LLMASM

**Status:** Draft plan — not yet implemented  
**Goal:** Make LLMASM efficient and robust when running on local LLMs with small context windows, limited instruction-following capacity, and slower inference. The plan combines ideas from `LiteLLM`'s context utilities, `hermes-agent`'s compression discipline, and local-model-specific adaptations.

This plan complements:
- `hermes-agent-execution-practices-plan.md` — execution hardening
- `hermes-litellm-tools-provider-plan.md` — tool and provider layer improvements

---

## 1. Current gaps in LLMASM

| Gap | Impact on local models |
|---|---|
| Default tokenizer is `ceil(words * 1.35)` heuristic | Inaccurate budgets, especially for code/multilingual text |
| No prompt overhead accounting | System prompt, JSON keys, tool schemas eat into context budget unnoticed |
| `default_context_tokens=4096` ignores actual model window | 2K–4K local models are easily overrun |
| No output-token reservation | Model may not have room to answer after a large context |
| LLM relevance filter is expensive and risky | Extra local LLM call per model node; small model may hallucinate filter IDs |
| Embeddings off by default | Retrieval falls back to brittle word overlap |
| Chunk size not model-aware | 256-token chunks can be too large for small windows |
| Artifact context is raw JSON | Tool/model artifacts are verbose in context |
| Planner context budget is static 800 tokens | Does not adapt to model window or planner output needs |
| No context-window pre-check | Requests are sent first, fail later |

---

## 2. Context-optimization best practices to import

### From LiteLLM

| Practice | Application to LLMASM |
|---|---|
| `trim_messages(model, max_tokens, trim_ratio=0.75)` | Trim context items to a target ratio of model window before building the prompt |
| `compress()` (BM25 + embedding relevance, retrieval tool) | Compress long context into stubs + `retrieve_content` tool; model pulls full text only when needed |
| Prompt caching (`cache_control`) | Keep stable system prompt / tool schemas warm across turns (for providers that support it) |
| `supports_prompt_caching()`, `get_supported_openai_params()` | Only enable caching/tool-calling if the local backend supports it |
| Context-window pre-check (`enable_pre_call_checks`) | Fail fast or fallback before sending an oversized prompt |
| `modify_params` / message sanitization | Repair orphaned tool calls/results and empty content before provider call |

### From hermes-agent

| Practice | Application to LLMASM |
|---|---|
| Iterative context compression with protected head/tail | Keep system prompt + recent turns + active task; summarize middle |
| Stable system-prompt snapshot | Persist and reuse system prompt across turns; reduce re-rendering |
| Tool availability gating | Only include tools that can actually run, reducing schema size |
| Tool description progressive disclosure | Replace rarely-used tools with `tool_search`/`tool_describe` bridge |
| Untrusted result wrapping | Mark web/MCP content so model does not confuse it with instructions |

### Local-model-specific

| Practice | Why it matters |
|---|---|
| Model-window-aware budgets | Local models often have 2K–8K windows; budgets must derive from `ModelInfo.context_window` |
| Real tokenizer integration | Heuristics cause over/under-counting; use Hugging Face / llama.cpp tokenizer |
| Small-model preset | Lower all budgets, smaller chunks, disable expensive filters, simpler tool schemas |
| Reserve output headroom | Ensure `context + max_output < model_window` |
| Shorter tool descriptions | Local models struggle with long tool schemas; prefer concise descriptions |
| Fewer tools per call | Send only the most relevant toolset, not the full catalog |

---

## 3. Proposed implementation phases

### Phase A: Accurate token accounting

1. **Ship real tokenizer adapters**
   - Add `HuggingFaceTokenizer` and `LlamaCppTokenizer` implementations of `llmasm.tokenizers.Tokenizer`.
   - Make tokenizer pluggable via `RuntimeConfig.tokenizer`.
   - Keep `SimpleTokenizer` as fallback.

2. **Prompt accounting utility**
   - Create `_prompt_budget(model_info, instruction, inputs, output_reserve, overhead)` that returns safe context token budget.
   - Overhead includes JSON framing, system prompt, tool schemas, and `--- Context ---` formatting.

3. **Replace word-based counts**
   - Update `InMemoryStorage.retrieve_workspace_context` and `PostgresStorage` to use `runtime_config.tokenizer` instead of `len(text.split())`.

4. **Output reservation**
   - Add `output_reserve_tokens` to `RuntimeConfig` (default 512).
   - Enforce: `context_budget = model_window - output_reserve - prompt_overhead`.

### Phase B: Model-window-aware budgets

1. **Read `ModelInfo.context_window` in runtime**
   - `select_context()` should use `min(default_context_tokens, model_window - output_reserve - overhead)`.

2. **Planner budget adaptation**
   - `_prior_context()` should reserve space for schema descriptions, tools, and planner output.
   - Formula: `prior_context_budget = min(800, planner_max_tokens - repair_reserve - tools_overhead)`.

3. **Fast-path instruction capping**
   - Trim or warn when fast-path instruction exceeds safe budget.

4. **Pre-call context-window check**
   - Before provider call, estimate total prompt tokens and raise/fallback if over model window.

### Phase C: Smarter context selection

1. **Enable embeddings by default when available**
   - Auto-enable `chat_embeddings_enabled` when an embedding model is configured.
   - Fall back to word overlap if embedding call fails.

2. **Graph-neighborhood boosting**
   - Follow `FOLLOWS_UP` / `PRODUCED` workspace edges to include directly related memory items.
   - Add provenance edge `USED_CONTEXT` so previously useful items get higher rank.

3. **Artifact summarization**
   - Store a text summary alongside each artifact.
   - Use summary in context; include raw JSON only if the node explicitly asks for it.

4. **Disable or simplify LLM relevance filter**
   - For small models, default `llm_context_filter=False`.
   - If enabled, cache results per query and cap candidates lower (e.g., 6 instead of 12).

### Phase D: Compression and chunking

1. **Model-aware chunk presets**
   - Add `small_model_chunk_preset` in `RuntimeConfig`: 128-token chunks, 16-token overlap for <4K models.
   - Auto-select preset based on `ModelInfo.context_window`.

2. **Chunk adjacency hints**
   - Preserve original ordering in context.
   - Add metadata like "Passage 2 of 5" so the model understands fragmentation.

3. **Hierarchical summarization**
   - For very large documents, build a tree: chunk summaries → section summaries → document summary.
   - Use summary-first retrieval.

4. **Prompt compression (LiteLLM-style)**
   - When context exceeds a trigger, compress middle items into stubs.
   - Inject a `retrieve_content` tool so the model can request full text on demand.
   - Cache stub → full-content mapping in the workspace.

### Phase E: Tool schema and prompt caching

1. **Tool schema minimization**
   - Strip `description` down to first sentence for small-model mode.
   - Remove `examples`, `default`, and long enums when over budget.
   - Use toolsets to limit tools per turn.

2. **Stable system prompt caching**
   - Persist system prompt snapshot per workspace/run.
   - Add `cache_control` breakpoints for providers that support prompt caching.
   - Use the snapshot hash as a cache key.

3. **Progressive tool disclosure**
   - When tool schema size exceeds a threshold, replace non-core tools with `tool_search`/`tool_describe`.

---

## 4. New `RuntimeConfig` fields

```python
@dataclass
class RuntimeConfig:
    # Existing fields...

    # Token accounting
    tokenizer: Tokenizer = field(default_factory=SimpleTokenizer)
    output_reserve_tokens: int = 512
    prompt_overhead_tokens: int = 256

    # Model-aware budgets
    auto_context_budget: bool = True  # derive from ModelInfo.context_window
    small_model_context_window_threshold: int = 4096
    small_model_preset: str = "small"  # selects chunk/tool schema presets

    # Context selection
    chat_embeddings_enabled: bool = True  # auto-enable if embedding model configured
    graph_neighborhood_boost: bool = True
    artifact_summaries_enabled: bool = True
    llm_context_filter: bool = False  # default off for local models

    # Chunking
    chunk_preset: str = "auto"  # "auto" | "small" | "medium" | "large"
    chunk_adjacency_hints: bool = True

    # Compression
    compression_enabled: bool = False
    compression_trigger_tokens: int = 4096
    compression_target_tokens: int = 2048

    # Tool schemas
    tool_schema_minimal_mode: bool = False  # small-model mode
    max_tools_per_turn: int | None = None
    progressive_tool_disclosure_threshold_tokens: int | None = None

    # Caching
    prompt_cache_enabled: bool = True
    prompt_cache_min_tokens: int = 1024
```

---

## 5. Integration with existing plans

| Existing plan | How this plan extends it |
|---|---|
| `hermes-agent-execution-practices-plan.md` | Iterative context compression, stable system prompt, message integrity repair, and config knobs are shared; this plan adds local-model-specific sizing |
| `hermes-litellm-tools-provider-plan.md` | Tool schema minimization and progressive disclosure build on the toolset/registry refactor; LiteLLM provider integration enables capability-aware context decisions |

---

## 6. Testing strategy

- Add `tests/unit/test_token_accounting.py` with mocked tokenizers and model info.
- Add `tests/unit/test_context_selection.py` for neighborhood boosting and artifact summaries.
- Add `tests/unit/test_chunking_presets.py` for model-aware chunk sizes.
- Add `tests/unit/test_prompt_compression.py` for stub + retrieve loop.
- Keep `tests/unit/test_v0.py` green.
- Run `make test`, `make lint`, `make typecheck`.

---

## 7. Open questions

1. **Tokenizer dependency:** Should Hugging Face tokenizers be an optional extra (`llmasm[hf-tokenizers]`) or a core dependency?
2. **Embedding model auto-detection:** Should LLMASM try Ollama's default embedding model (`nomic-embed-text`) automatically, or require explicit config?
3. **Compression stub tool:** Should the `retrieve_content` compression tool be a real workspace tool, or an internal executor mechanism?
4. **Prompt caching:** Should we implement cache-control breakpoints only for cloud providers, or also investigate llama.cpp/Ollama KV-cache reuse?
5. **Small-model preset thresholds:** Is 4096 tokens the right threshold for "small model"? Should it be 8192?
6. **Artifact summaries:** Should summaries be generated eagerly on artifact creation, or lazily when first used in context?

---

## References

- `llmasm/runtime/context.py`
- `llmasm/chunking.py`
- `llmasm/tokenizers.py`
- `llmasm/config.py`
- `llmasm/runtime/executor.py`
- `llmasm/compiler/compiler.py`
- `llmasm/storage/embeddings.py`
- `llmasm/conversation/retrieval.py`
- `llmasm/conversation/memory.py`
- `llmasm/conversation/chat.py`
- `llmasm/providers/base.py`
- LiteLLM docs: `trim_messages`, `compress()`, prompt caching, `modify_params`
