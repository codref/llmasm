# Plan: Unlock LLMASM’s Graph Potential

**Status:** Draft plan — not yet implemented  
**Goal:** Move LLMASM from using its graph mainly as an execution DAG and storage schema to exploiting the persistent typed graph as a first-class reasoning, retrieval, provenance, and optimization substrate.

This plan complements:
- `hermes-agent-execution-practices-plan.md` — execution hardening
- `hermes-litellm-tools-provider-plan.md` — tool and provider layer improvements
- `local-first-context-optimization-plan.md` — context/token efficiency for local models

---

## 1. Current state of graph usage

### What is already used well

- `TaskGraph` + `TaskEdge` DAG execution: the scheduler and executor rely on task edges for dataflow, routing, skipping, and caching.
- `WorkspaceEdge` schema and persistence: `supports_goal` and `expands_to` edges are written by the compiler and expansion module.
- Typed ports/schemas and transforms are validated and applied during input gathering.
- Embeddings + word-overlap memory retrieval are wired into `select_context`.

### What is defined but dormant or underused

| Feature | Where defined | Current state |
|---|---|---|
| `produced`, `used_context`, `summarizes`, `refers_to`, `contradicts` edge types | `llmasm/graph/models.py` | Almost never written or queried |
| `memory_query`, `observation`, `goal` node kinds | `llmasm/graph/models.py` | Raise `ExecutionError` |
| `TaskGraph.parent_task_graph_id` | model + DB schema | Never populated |
| Apache AGE graph backend | Docker Postgres `initdb`, migration | No Python code queries it |
| Graph topology in context selection | `select_context()` | Not used; only vector/text overlap |
| Cross-run workspace visualization | `analysis/visualize.py` | Only one task graph/run at a time |
| Structural few-shot/template reuse | storage `retrieve_few_shot_examples` | Only intent word overlap |

---

## 2. Opportunities

### 2.1 Graph-aware context selection (highest impact)

**Idea:** Walk workspace edges from the current node/task graph to find structurally relevant artifacts, not just text-similar ones.

**How:**
- Add `Storage.traverse_workspace(start_ids, edge_types, max_depth)` to the `Storage` protocol.
- In `select_context`, after vector/text retrieval, traverse `follows_up`, `supports_goal`, `refers_to`, and `used_context` edges up to depth 2–3.
- Merge graph-derived candidates with semantic candidates and re-rank before token trimming.

**Why:** Local models benefit from explicit provenance links. A follow-up question is structurally related to its parent even if wording differs.

**Complexity:** Medium.

### 2.2 Persist provenance edges

**Idea:** Every execution step should write explicit provenance into the workspace graph.

**How:**
- `produced`: node → artifact.
- `used_context`: model/compress node → context items used.
- `summarizes`: summary memory item → source artifact.
- `refers_to`: answer/question → source passages.

**Why:** Enables attribution, debugging, and powers graph traversal.

**Complexity:** Medium.

### 2.3 Implement `memory_query`, `observation`, and `goal` node kinds

**Idea:** Make graph memory and goals first-class runtime primitives.

**How:**
- `memory_query`: structured query against workspace graph (vector + edge traversal + filters).
- `observation`: assert a fact, persist as `MemoryItem`, link via `refers_to`/`produced`.
- `goal`: executable objective node resolved against `GoalTracker`.

**Why:** Moves LLMASM from “graph stores execution trace” to “graph is the reasoning substrate.”

**Complexity:** Medium–High.

### 2.4 Hierarchical task graphs via `parent_task_graph_id`

**Idea:** Link follow-up task graphs to their parent.

**How:**
- Set `parent_task_graph_id` when `goal_action` is `continue` or `steer`.
- Add `Storage.load_task_graph_chain()` and use it for context + analysis.

**Why:** Cheap change that makes follow-ups and run analysis much richer.

**Complexity:** Low–Medium.

### 2.5 Apache AGE graph backend

**Idea:** Use AGE/Cypher for fast, expressive graph traversal over workspace edges.

**How:**
- Mirror `workspace_edges` and `task_edges` into AGE vertices/edges on write.
- Implement traversal methods (`follow_up_chain`, `expansion_tree`, `goal_subgraph`) with Cypher, falling back to recursive SQL.

**Why:** Graph queries over recursive SQL; enables complex analysis and visualization.

**Complexity:** High.

### 2.6 Graph-based few-shot / template reuse

**Idea:** Retrieve prior task graphs by structural similarity, not just intent overlap.

**How:**
- Compute structural fingerprints: node kinds, tool names, schema flow.
- Rank few-shot examples by combined intent + structure score.
- Extract recurring subgraph motifs as named templates.

**Why:** Planner gets reusable recipes for common local workflows.

**Complexity:** Medium.

### 2.7 Contradiction detection and belief revision

**Idea:** Detect conflicting memory items and surface them.

**How:**
- On new `observation`/fact, compare embedding + LLM against existing items.
- Write `contradicts` edges.
- Add policy: pause downstream nodes or route to resolution.

**Why:** Turns memory from append-only bag to a belief graph.

**Complexity:** Medium.

### 2.8 Subgraph-aware artifact cache

**Idea:** Reuse artifacts from equivalent upstream subgraphs across runs.

**How:**
- Compute canonical subgraph hash from node kinds, execution, and topology.
- Index artifacts by subgraph hash.
- In executor, check for equivalent subgraph before invoking.

**Why:** Cross-run optimization; local models re-execute expensive tools less often.

**Complexity:** Medium–High.

### 2.9 Workspace-level visualization and analysis

**Idea:** Export the full workspace graph across all runs, not just one task graph.

**How:**
- Add `query_workspace(workspace_graph_id)`.
- Extend `to_viewer_graph` with heterogeneous nodes and workspace edges.
- Expose `LLMASM.query_workspace()`.

**Why:** Quick win that makes the graph tangible and debuggable.

**Complexity:** Low–Medium.

---

## 3. Proposed implementation phases

### Phase 1 — Foundations (enable everything)

1. **Workspace-level visualization**  
   Export full workspace graph with heterogeneous nodes and workspace edges. This makes the graph concrete and helps debug later phases.

2. **Provenance edges**  
   Write `produced`, `used_context`, `summarizes`, and `refers_to` edges during execution, chunking, and retrieval.

3. **Hierarchical task graphs**  
   Populate `parent_task_graph_id` for `continue`/`steer` actions and expose ancestor chains.

### Phase 2 — Graph-powered retrieval

4. **Graph-aware context selection**  
   Add `Storage.traverse_workspace()` and use workspace edges to seed context retrieval.

5. **`memory_query` node kind**  
   Implement structured workspace-memory queries as an executable node.

### Phase 3 — Graph-powered reasoning

6. **`observation` and `goal` node kinds**  
   Make fact assertions and objectives executable graph citizens.

7. **Contradiction detection**  
   Compare new observations against existing memory and write `contradicts` edges.

8. **Graph-based few-shot / templates**  
   Retrieve prior graphs by structural similarity and extract reusable subgraph templates.

### Phase 4 — Optimization and advanced backend

9. **Subgraph-aware artifact cache**  
   Reuse artifacts across runs when upstream subgraphs are equivalent.

10. **Apache AGE backend**  
    Mirror edges into AGE and implement Cypher traversations with SQL fallback.

---

## 4. New `RuntimeConfig` fields

```python
@dataclass
class RuntimeConfig:
    # Existing fields...

    # Graph traversal for context
    graph_context_traversal_enabled: bool = True
    graph_context_max_depth: int = 2
    graph_context_edge_types: list[str] = field(
        default_factory=lambda: ["follows_up", "supports_goal", "refers_to", "used_context"]
    )
    graph_context_boost_score: float = 0.2

    # Provenance
    provenance_edges_enabled: bool = True

    # Hierarchical graphs
    link_parent_task_graphs: bool = True

    # Advanced nodes
    memory_query_enabled: bool = True
    observation_enabled: bool = True
    goal_nodes_enabled: bool = True

    # Contradiction detection
    contradiction_detection_enabled: bool = False
    contradiction_similarity_threshold: float = 0.85

    # Few-shot / templates
    structural_few_shot_enabled: bool = False
    subgraph_template_mining_enabled: bool = False

    # Subgraph cache
    subgraph_cache_enabled: bool = True

    # AGE backend
    age_enabled: bool = False  # auto-detect if AGE extension present
```

---

## 5. Storage protocol additions

Add these methods to `Storage` (and implement in `InMemoryStorage` + `PostgresStorage`):

```python
def traverse_workspace(
    self,
    workspace_graph_id: str,
    start_ids: list[str],
    edge_types: list[str] | None,
    max_depth: int,
    direction: str = "out",
) -> list[tuple[str, str, str]]: ...  # (source_id, edge_type, target_id)

def load_task_graph_chain(
    self, task_graph_id: str
) -> list[TaskGraph]: ...

def query_workspace(
    self, workspace_graph_id: str
) -> WorkspaceGraph: ...

def find_equivalent_subgraph_hash(
    self, subgraph_hash: str
) -> str | None: ...

def write_provenance_edge(
    self, edge: WorkspaceEdge
) -> None: ...
```

---

## 6. Testing strategy

- Add `tests/unit/test_graph_traversal.py` for workspace-edge traversal.
- Add `tests/unit/test_provenance_edges.py` for `produced`, `used_context`, etc.
- Add `tests/unit/test_hierarchical_task_graphs.py` for parent linking.
- Add `tests/unit/test_memory_query_node.py` and `tests/unit/test_observation_node.py`.
- Add `tests/unit/test_workspace_visualization.py` for full-workspace export.
- Keep `tests/unit/test_v0.py` green.
- Run `make test`, `make lint`, `make typecheck`.

---

## 7. Open questions

1. **Apache AGE priority:** Is AGE a hard requirement, or should all traversal also work on plain Postgres (recursive CTEs) and `InMemoryStorage`?
2. **`memory_query` scope:** Should it be a planner-emitted node, an internal executor helper, or both?
3. **`observation` authority:** Should any tool/model output be able to create observations, or only explicitly marked nodes?
4. **Contradiction resolution:** Should conflicts block execution, or just produce a warning/router branch?
5. **Visualization format:** Enhance the existing JSON viewer, or add Mermaid/DOT export?
6. **Order:** Should we start with Phase 1 as written, or jump straight to graph-aware context selection (2.1) first?

---

## References

- `llmasm/graph/models.py`
- `llmasm/runtime/context.py`
- `llmasm/runtime/executor.py`
- `llmasm/runtime/scheduler.py`
- `llmasm/storage/base.py`
- `llmasm/storage/postgres.py`
- `llmasm/storage/memory.py`
- `llmasm/storage/migrations/001_initial.sql`
- `llmasm/compiler/compiler.py`
- `llmasm/analysis/visualize.py`
- `llmasm/analysis/run.py`
- `llmasm/conversation/memory.py`
- `llmasm/conversation/retrieval.py`
- `llmasm/goals/tracker.py`
- `docker/postgres/initdb/001_extensions.sql`
