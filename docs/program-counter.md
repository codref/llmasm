# Program Counter

## What it is

`program_counter_node_id` is a single nullable field that appears on two models:

- `Run` — the live value, updated before every node execution
- `Checkpoint` — a snapshot of that value frozen at each checkpoint moment

It always holds the `id` of the **node currently being (or last) executed** in the run.

---

## Where it lives in the schema

```python
class Run(BaseModel):
    id: str
    workspace_graph_id: str
    task_graph_id: str
    status: RunStatus = RunStatus.PENDING
    program_counter_node_id: str | None = None   # ← PC
    ...

class Checkpoint(BaseModel):
    id: str
    run_id: str
    program_counter_node_id: str | None = None   # ← snapshot of PC
    completed_node_ids: list[str]
    failed_node_ids: list[str]
    state_hash: str
    ...
```

---

## When it moves

The executor advances the PC in a tight bracket around every node:

```
run.program_counter_node_id = node.id   # PC advances to this node
storage.update_run(run)                 # written to DB immediately

_checkpoint(run)                        # checkpoint A  ← PC = this node, node not yet run

_execute_node(run, node)                # node runs here

_checkpoint(run)                        # checkpoint B  ← PC = this node, node just finished
```

After checkpoint B the loop calls `scheduler.next_node` again and the PC
advances to whatever the scheduler returns next.

---

## What the PC is NOT

The program counter does **not** drive execution order.

The `Scheduler` computes the executable frontier entirely from `RunNodeState`
rows.  For every node whose `status == PENDING` it checks whether all upstream
nodes (connected by `TaskEdge`) are in `{SUCCEEDED, EXPANDED}`.  The first
unblocked node is returned.  The PC is invisible to the scheduler.

---

## What the PC is for

### 1. Live observability

While a run is in progress, reading `runs.program_counter_node_id` tells you
exactly which node the executor is working on right now.

```sql
SELECT r.status, n.name AS current_node, r.program_counter_node_id
FROM runs r
LEFT JOIN nodes n ON n.id = r.program_counter_node_id
WHERE r.id = '<run_id>';
```

### 2. Failure diagnosis

When a run ends with `status = failed`, the PC points at the node that was
executing when the exception was raised.

```sql
SELECT r.status, n.name AS failed_at, r.program_counter_node_id
FROM runs r
JOIN nodes n ON n.id = r.program_counter_node_id
WHERE r.status = 'failed';
```

### 3. Checkpoint attribution

Each run produces two `Checkpoint` rows per node (one before, one after).
Because both checkpoints record the same `program_counter_node_id`, you can
reconstruct the exact before/after bracket for any node:

```sql
SELECT c.created_at, c.state_hash, n.name AS pc_node
FROM checkpoints c
JOIN nodes n ON n.id = c.program_counter_node_id
WHERE c.run_id = '<run_id>'
ORDER BY c.created_at;
```

### 4. Resume on crash (future)

A recovery path would:
1. Read the last checkpoint's `program_counter_node_id` and `completed_node_ids`.
2. Replay those node states as `SUCCEEDED`.
3. Re-run from the PC node forward.

This is not implemented in v0 but the data is already captured.

---

## Analogy and key distinction

In a CPU, the program counter advances through a flat array of instructions in
sequential order — it both tracks and drives execution.

In llmasm the task graph is a **DAG**.  The `Scheduler` drives execution order
through topological analysis of `RunNodeState` and `TaskEdge` rows.  The PC is
a **breadcrumb written onto the `Run` row** — it records *where the executor
is*, but it has no influence over *what runs next*.
