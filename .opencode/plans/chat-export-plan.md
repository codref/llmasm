# Plan: Add `--output-file` and `/export` to chat.py

## Goal
Add a logging mechanism to `examples/chat.py` that stores Q&A pairs as JSONL, enabling downstream validation of answer quality and coherence.

## Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Format | JSONL | Streaming-friendly, easy to append, and simple for downstream tools |
| `--output-file` behavior | Append after each turn | Unattended mode should preserve partial results on crash |
| `/export` behavior | Overwrite target file | Explicit command means user wants a fresh snapshot |
| `/export` without args | Uses `--output-file` path if set | Convenient for users who started with `--output-file` |
| Failed turns | Logged in `--input-file` mode | Quality validation needs to see failures too |
| REPL turns | Only exported via `/export` | Avoids surprise disk writes during casual chat |

## JSONL Record Schema

```json
{"question": "...", "answer": "...", "goal_action": "...", "status": "succeeded", "task_graph_id": "...", "run_id": "...", "error": null}
```

- `question`: User prompt
- `answer`: Final answer text (empty if no output or failure)
- `goal_action`: From task graph metadata (empty if compilation failed)
- `status`: `succeeded` or `failed` (or `cancelled` from Run status)
- `task_graph_id`: ID of the compiled task graph
- `run_id`: ID of the execution run
- `error`: Human-readable error string on failure, `null` on success

## Implementation Details

### 1. New imports
Add `import json` to the top of `examples/chat.py`.

### 2. New helper functions

```python
def _parse_qa(text: str) -> tuple[str, str]:
    """Parse a MemoryItem text in 'Q: ...\\nA: ...' format."""
    if text.startswith("Q: "):
        parts = text.split("\n", 1)
        question = parts[0][3:]
        if len(parts) > 1 and parts[1].startswith("A: "):
            answer = parts[1][3:]
        else:
            answer = ""
    else:
        question = text
        answer = ""
    return question, answer


def _write_jsonl_record(
    file_path: str,
    question: str,
    answer: str,
    goal_action: str,
    status: str,
    task_graph_id: str,
    run_id: str,
    error: str | None,
) -> None:
    """Append a single conversation turn as JSONL."""
    record = {
        "question": question,
        "answer": answer,
        "goal_action": goal_action,
        "status": status,
        "task_graph_id": task_graph_id,
        "run_id": run_id,
        "error": error,
    }
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _export_workspace_turns(
    file_path: str,
    workspace_id: str,
    storage: InMemoryStorage | PostgresStorage,
    console: Console,
) -> None:
    """Export all kind=turn memory items from workspace to JSONL (overwrite)."""
    items = storage.list_memory_items(workspace_id)
    turns = [item for item in items if item.kind == "turn"]
    turns.sort(key=lambda x: x.created_at)

    records = []
    for memory in turns:
        question, answer = _parse_qa(memory.text)
        goal_action = ""
        status = "unknown"
        task_graph_id = ""
        run_id = memory.source_run_id or ""

        if memory.source_run_id:
            try:
                run = storage.load_run(memory.source_run_id)
                status = run.status
                task_graph_id = run.task_graph_id
                if run.task_graph_id:
                    task_graph = storage.load_task_graph(run.task_graph_id)
                    goal_action = task_graph.metadata.get("goal_action", "")
            except Exception:
                pass  # Graceful degradation if run/task_graph is missing

        records.append({
            "question": question,
            "answer": answer,
            "goal_action": goal_action,
            "status": status,
            "task_graph_id": task_graph_id,
            "run_id": run_id,
            "error": None,
        })

    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    console.print(f"[dim]Exported {len(records)} turn(s) to {file_path}[/dim]")
```

### 3. Argument parser change

Add after the `--input-file` argument:

```python
    parser.add_argument(
        "--output-file",
        default=None,
        metavar="PATH",
        help="Path to append JSONL conversation log in --input-file mode, and default path for /export.",
    )
```

### 4. `--input-file` loop restructuring

Current loop uses `continue` in `except` blocks, which skips all post-processing. Restructure to capture failures and write them to the output file:

```python
    if args.input_file is not None:
        questions = Path(args.input_file).read_text().splitlines()
        questions = [q.strip() for q in questions if q.strip()]
        prev_task_graph_id: str | None = None
        prev_final_node_id: str | None = None
        for turn_idx, question in enumerate(questions):
            console.print(f"[dim][{turn_idx}]>[/dim] {question}")
            console.print()

            status = "succeeded"
            error: str | None = None
            task_graph_id: str | None = None
            run_id: str | None = None
            analysis: RunAnalysis | None = None
            answer = FinalAnswer(text="", sources=[])

            try:
                task_graph_id = app.compile(workspace_id, question)
                run_id = app.run(task_graph_id)
                analysis = app.query_run(run_id)
            except CompilationError as exc:
                status = "failed"
                error = f"Planner failed after {exc.attempts} attempt(s): {exc.last_errors}"
                console.print(f"[red]Planner failed after {exc.attempts} attempt(s).[/red]")
                if exc.last_errors:
                    console.print(f"[red]Last errors: {exc.last_errors}[/red]")
                prev_task_graph_id = None
                prev_final_node_id = None
                console.print()
            except LLMASMError as exc:
                status = "failed"
                error = str(exc)
                console.print(f"[red]Error: {exc}[/red]")
                prev_task_graph_id = None
                prev_final_node_id = None
                console.print()

            if analysis:
                final_id = _final_node_id(analysis)
                answer = _extract_answer(analysis, final_id)

                if answer.text:
                    console.print(Markdown(answer.text))
                    if answer.sources:
                        console.print(f"[dim]sources: {answer.sources}[/dim]")
                else:
                    console.print("[dim](no output)[/dim]")

                goal_action = analysis.task_graph.metadata.get("goal_action", "")
                tg_short = task_graph_id.split("_")[-1][:12]
                console.print(f"[dim]── turn {turn_idx}  |  goal: {goal_action}  |  tg: {tg_short}[/dim]")

                memory = write_memory_item(
                    workspace_graph_id=workspace_id,
                    kind="turn",
                    text=f"Q: {question}\nA: {answer.text}",
                    runtime_config=app.runtime_config,
                    provider=app.provider,
                    embedding_store=app.embedding_store,
                    storage=storage,
                    source_run_id=run_id,
                )

                if prev_task_graph_id is not None:
                    storage.persist_workspace_edge(
                        WorkspaceEdge(
                            id=new_id("edge"),
                            workspace_graph_id=workspace_id,
                            edge_type=WorkspaceEdgeType.FOLLOWS_UP,
                            from_type="task_graph",
                            from_id=task_graph_id,
                            to_type="task_graph",
                            to_id=prev_task_graph_id,
                            reason=f"turn {turn_idx} follows turn {turn_idx - 1}",
                        )
                    )

                if final_id is not None:
                    storage.persist_workspace_edge(
                        WorkspaceEdge(
                            id=new_id("edge"),
                            workspace_graph_id=workspace_id,
                            edge_type=WorkspaceEdgeType.PRODUCED,
                            from_type="node",
                            from_id=final_id,
                            to_type="memory_item",
                            to_id=memory.id,
                            reason="final node answer stored as workspace memory",
                        )
                    )

                prev_task_graph_id = task_graph_id
                prev_final_node_id = final_id
                console.print()

            if args.output_file:
                _write_jsonl_record(
                    file_path=args.output_file,
                    question=question,
                    answer=answer.text,
                    goal_action=analysis.task_graph.metadata.get("goal_action", "") if analysis else "",
                    status=status,
                    task_graph_id=task_graph_id or "",
                    run_id=run_id or "",
                    error=error,
                )
        return
```

### 5. REPL `/export` commands

Add after `/run` command handling:

```python
        if text == "/export":
            if not args.output_file:
                console.print("[red]No default output file set. Use /export <path> or start with --output-file.[/red]")
                continue
            _export_workspace_turns(args.output_file, workspace_id, storage, console)
            continue

        if text.startswith("/export "):
            path = text[8:].strip()
            if path:
                _export_workspace_turns(path, workspace_id, storage, console)
            continue
```

### 6. `/help` update

Add `/export` to the help text:

```python
                "  /export          — export all turns to the default output file (requires --output-file)\n"
                "  /export <path>   — export all turns to the specified JSONL file\n"
```

## Files to modify

- `examples/chat.py` (all changes)

## Testing approach

1. Create a test input file with 3 questions
2. Run `python examples/chat.py --input-file test.txt --output-file test.jsonl`
3. Verify `test.jsonl` contains 3 lines (or 3 lines if all succeed, fewer if some fail)
4. Run `python examples/chat.py` (interactive mode), ask a question, then type `/export /tmp/export.jsonl`
5. Verify `/tmp/export.jsonl` contains the turn

## Risk assessment

- **Low risk**: Changes are localized to `examples/chat.py` (a utility script, not core library)
- **No API changes**: Only adds new CLI flags and commands
- **Backward compatible**: `--output-file` is optional; existing behavior unchanged when absent
- **No test impact**: `examples/chat.py` has no direct unit tests in the test suite
