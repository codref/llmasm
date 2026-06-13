"""Minimal chat interface for llmasm.

Dependencies (opt-in, install with `pip install llmasm[chat]`):
  prompt-toolkit>=3  – styled REPL prompt with input history
  rich>=13           – Markdown rendering for model responses

Try:
    python examples/chat.py
    python examples/chat.py --runtime-model gemma4:e4b --context-turns 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmasm.api import LLMASM
from llmasm.analysis.run import RunAnalysis
from llmasm.config import RuntimeConfig
from llmasm.errors import CompilationError, LLMASMError
from llmasm.goals.tracker import GoalTracker
from llmasm.graph.models import MemoryItem, WorkspaceEdge, WorkspaceEdgeType, WorkspaceGraph
from llmasm.graph.registry import default_schema_registry
from llmasm.ids import new_id
from llmasm.providers.ollama import OllamaProvider
from llmasm.schemas import FinalAnswer
from llmasm.storage.embeddings import InMemoryEmbeddingStore, NullEmbeddingStore, write_memory_item
from llmasm.storage.memory import InMemoryStorage
from llmasm.storage.postgres import PostgresEmbeddingStore, PostgresStorage
from llmasm.tools.reset_embeddings import reset_embeddings


STYLE = Style.from_dict(
    {
        "prompt": "#00aa00 bold",
        "separator": "dim",
    }
)


def _final_node_id(analysis: RunAnalysis) -> str | None:
    """Return the ID of the final node in the task graph, if any."""
    for node in analysis.task_graph.nodes:
        if node.kind == "final":
            return node.id
    return None


def _extract_answer(analysis: RunAnalysis, final_id: str | None) -> FinalAnswer:
    """Extract the FinalAnswer artifact produced by the final node."""
    if final_id is None:
        return FinalAnswer(text="", sources=[])
    artifacts = [a for a in analysis.artifacts if a.node_id == final_id]
    if not artifacts or not artifacts[-1].content_json:
        return FinalAnswer(text="", sources=[])
    return FinalAnswer.model_validate(artifacts[-1].content_json)


def _parse_qa(text: str) -> tuple[str, str]:
    """Parse a MemoryItem text in 'Q: ...\nA: ...' format."""
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
    context_length: int | None = None,
    search_query: str | None = None,
    rag_enabled: bool = False,
) -> None:
    """Append a single conversation turn as JSONL."""
    record: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "goal_action": goal_action,
        "status": status,
        "task_graph_id": task_graph_id,
        "run_id": run_id,
        "error": error,
        "rag_enabled": rag_enabled,
    }
    if context_length is not None:
        record["context_length"] = context_length
    if search_query:
        record["search_query"] = search_query
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

        records.append(
            {
                "question": question,
                "answer": answer,
                "goal_action": goal_action,
                "status": status,
                "task_graph_id": task_graph_id,
                "run_id": run_id,
                "error": None,
            }
        )

    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    console.print(f"[dim]Exported {len(records)} turn(s) to {file_path}[/dim]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat with llmasm")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--planner-model", default="llama3.1:8b")
    parser.add_argument("--runtime-model", default="llama3.1:8b")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--embedding-dimensions", type=int, default=768)
    parser.add_argument(
        "--embeddings",
        action="store_true",
        default=False,
        help="Enable vector embeddings for context retrieval",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--compiler-attempts", type=int, default=3)
    parser.add_argument("--workspace-name", default="chat", help="Persistent workspace name")
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Start a new isolated session (appends a timestamp to the workspace name)",
    )
    parser.add_argument(
        "--ref-workspace",
        action="append",
        dest="ref_workspaces",
        default=[],
        metavar="NAME",
        help="Read-only workspace to include as extra context source (repeatable)",
    )
    parser.add_argument(
        "--context-turns",
        type=int,
        default=10,
        help="Max MemoryItems retrieved as context per turn (0 = unlimited)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        metavar="DSN",
        help="PostgreSQL DSN for persistent storage, e.g. postgresql://llmasm:llmasm@localhost:15432/llmasm",
    )
    parser.add_argument(
        "--llm-goal-classifier",
        action="store_true",
        default=False,
        help="Use the planner model to classify goal actions instead of heuristics",
    )
    parser.add_argument(
        "--classifier-context-depth",
        type=int,
        default=3,
        help="Number of recent workspace memory items to feed into the goal classifier (0 = disabled)",
    )
    parser.add_argument(
        "--classifier-goal-text-chars",
        type=int,
        default=400,
        help="Max characters of the active goal text shown to the goal classifier",
    )
    parser.add_argument(
        "--llm-context-filter",
        action="store_true",
        default=False,
        help="Use the runtime model to filter retrieved context items for relevance before each node",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        metavar="PATH",
        help="Path to a file with one question per line. Runs all turns then exits (no REPL).",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        metavar="PATH",
        help="Path to append JSONL conversation log (auto-logs in --input-file mode and REPL mode).",
    )
    parser.add_argument(
        "--fast-path",
        action="store_true",
        default=False,
        help="Use deterministic conversation fast path (intent -> model -> final) instead of planner",
    )
    parser.add_argument(
        "--chat-embeddings",
        action="store_true",
        default=False,
        help="Enable embeddings for the conversation fast path (required for RAG retrieval)",
    )
    parser.add_argument(
        "--chunking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically chunk long source passages in fast-path mode (default: on)",
    )
    parser.add_argument(
        "--chunking-trigger-tokens",
        type=int,
        default=512,
        metavar="N",
        help="Token threshold above which a source passage is chunked (default: 512)",
    )
    parser.add_argument(
        "--chunk-target-tokens",
        type=int,
        default=256,
        metavar="N",
        help="Target token size for each chunk (default: 256)",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=32,
        metavar="N",
        help="Overlap tokens between consecutive chunks (default: 32)",
    )
    parser.add_argument(
        "--chunking-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a summary node when chunking a source passage (default: on)",
    )
    parser.add_argument(
        "--llm-query-rewrite",
        action="store_true",
        default=False,
        help="Rewrite follow-up questions into standalone search queries via LLM (fast-path only)",
    )
    parser.add_argument(
        "--llm-dialogue-classifier",
        action="store_true",
        default=False,
        help="Use the runtime model to classify dialogue type instead of heuristics (fast-path only)",
    )
    parser.add_argument(
        "--qa-truncate-chars",
        type=int,
        default=None,
        metavar="N",
        help="Truncate assistant answers in Q/A context to N characters (fast-path only)",
    )
    parser.add_argument(
        "--reset-embeddings",
        action="store_true",
        default=False,
        help="Reset pgvector embeddings (drop vector column + clear table) before starting. "
        "Useful when switching to a different embedding model with a different dimension.",
    )
    return parser.parse_args()


def _build_storage(args: argparse.Namespace) -> InMemoryStorage | PostgresStorage:
    if args.db_url:
        return PostgresStorage(args.db_url)
    return InMemoryStorage()


def _build_embedding_store(
    args: argparse.Namespace,
    storage: InMemoryStorage | PostgresStorage,
) -> InMemoryEmbeddingStore | PostgresEmbeddingStore | NullEmbeddingStore:
    # --chat-embeddings implies we need a real embedding store for fast-path RAG
    needs_embeddings = args.embeddings or args.chat_embeddings
    if not needs_embeddings:
        return NullEmbeddingStore()
    if args.db_url and isinstance(storage, PostgresStorage):
        return PostgresEmbeddingStore.create(storage.conn, dimensions=args.embedding_dimensions)
    return InMemoryEmbeddingStore()


def _stable_workspace_id(name: str) -> str:
    """Deterministic workspace ID derived from the workspace name for persistent backends."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "default"
    return f"workspace_{slug}"


def build_app(args: argparse.Namespace) -> LLMASM:
    provider = OllamaProvider(
        base_url=args.ollama_url,
        timeout=args.timeout,
        default_model=args.runtime_model,
        embedding_model=args.embedding_model,
    )
    storage = _build_storage(args)
    embedding_store = _build_embedding_store(args, storage)
    ref_ids = [_stable_workspace_id(name) for name in (args.ref_workspaces or [])]
    return LLMASM(
        storage=storage,
        provider=provider,
        runtime_config=RuntimeConfig(
            planner_model=args.planner_model,
            default_model=args.runtime_model,
            compiler_max_attempts=args.compiler_attempts,
            embedding_model=args.embedding_model,
            embedding_dimensions=args.embedding_dimensions,
            embeddings_enabled=args.embeddings,
            reference_workspace_ids=ref_ids,
            llm_goal_classifier=args.llm_goal_classifier,
            classifier_context_depth=args.classifier_context_depth,
            classifier_goal_text_chars=args.classifier_goal_text_chars,
            llm_context_filter=args.llm_context_filter,
            llm_query_rewrite=args.llm_query_rewrite,
            llm_dialogue_classifier=args.llm_dialogue_classifier,
            chat_qa_truncate_chars=args.qa_truncate_chars,
            chat_embeddings_enabled=args.chat_embeddings,
            chunking_enabled=args.chunking,
            chunking_trigger_tokens=args.chunking_trigger_tokens,
            chunk_target_tokens=args.chunk_target_tokens,
            chunk_overlap_tokens=args.chunk_overlap_tokens,
            chunking_summary_enabled=args.chunking_summary,
        ),
        schema_registry=default_schema_registry(),
        embedding_store=embedding_store,
    )


def main() -> None:
    args = parse_args()
    console = Console()

    if args.reset_embeddings and args.db_url:
        reset_embeddings(args.db_url)

    if args.fresh and args.db_url:
        import time
        args.workspace_name = f"{args.workspace_name}_{int(time.time())}"

    app = build_app(args)
    storage = app.storage
    goal_tracker = GoalTracker(storage)

    if args.db_url:
        workspace_id = _stable_workspace_id(args.workspace_name)
        app.storage.create_workspace_graph(
            WorkspaceGraph(id=workspace_id, name=args.workspace_name)
        )
        storage_label = "[persistent: postgres]"
    else:
        workspace_id = app.create_workspace(args.workspace_name)
        storage_label = "[in-memory]"

    if args.embeddings:
        embedding_label = f"[embeddings: on ({args.embedding_model}, {args.embedding_dimensions}d)]"
    else:
        embedding_label = "[embeddings: off]"

    log_label = f"[logging: on → `{args.output_file}`]" if args.output_file else "[logging: off]"
    console.print(
        Markdown(
            f"**llmasm chat**  |  workspace: `{args.workspace_name}`  "
            f"|  model: `{args.runtime_model}`  |  planner: `{args.planner_model}`  "
            f"|  storage: `{storage_label}`  |  {embedding_label}  |  {log_label}\n"
            f"Type /help for commands, /quit or Ctrl-D to exit."
        )
    )
    console.print("─" * console.width, style="dim")

    # ── File-driven mode ────────────────────────────────────────────────────
    if args.input_file is not None:
        questions = Path(args.input_file).read_text().splitlines()
        questions = [q.strip() for q in questions if q.strip()]
        prev_task_graph_id: str | None = None

        for turn_idx, question in enumerate(questions):
            console.print(f"[dim][{turn_idx}]>[/dim] {question}")
            console.print()

            status = "succeeded"
            error: str | None = None
            task_graph_id: str | None = None
            run_id: str | None = None
            analysis: RunAnalysis | None = None
            answer = FinalAnswer(text="", sources=[])
            goal_action = ""

            context_length: int | None = None
            search_query: str | None = None
            rag_enabled = False

            if args.fast_path:
                turn_info: dict[str, Any] = {}
                try:
                    answer = app.chat(workspace_id, question, out_info=turn_info, turn=turn_idx)
                    context_length = turn_info.get("instruction_tokens")
                    run_id = turn_info.get("run_id", "")
                    search_query = turn_info.get("search_query")
                    rag_enabled = bool(turn_info.get("rag_enabled", False))
                except LLMASMError as exc:
                    status = "failed"
                    error = str(exc)
                    console.print(f"[red]Error: {exc}[/red]")
                if answer.text:
                    console.print(Markdown(answer.text))
                else:
                    console.print("[dim](no output)[/dim]")
                ctx_label = f"  |  ctx={context_length}tk" if context_length is not None else ""
                console.print(f"[dim]── turn {turn_idx}  |  [fast path]{ctx_label}[/dim]")
                console.print()
            else:
                try:
                    task_graph_id = app.compile(workspace_id, question)
                    run_id = app.run(task_graph_id)
                    analysis = app.query_run(run_id)
                    # Sum token counts of all prompt artifacts in the run
                    if analysis:
                        prompt_artifacts = [a for a in analysis.artifacts if a.port == "prompt"]
                        context_length = sum(a.token_count for a in prompt_artifacts) if prompt_artifacts else None
                except CompilationError as exc:
                    status = "failed"
                    error = f"Planner failed after {exc.attempts} attempt(s): {exc.last_errors}"
                    console.print(f"[red]Planner failed after {exc.attempts} attempt(s).[/red]")
                    if exc.last_errors:
                        console.print(f"[red]Last errors: {exc.last_errors}[/red]")
                    prev_task_graph_id = None
                    console.print()
                except LLMASMError as exc:
                    status = "failed"
                    error = str(exc)
                    console.print(f"[red]Error: {exc}[/red]")
                    prev_task_graph_id = None
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
                    ctx_label = f"  |  ctx={context_length}tk" if context_length is not None else ""
                    console.print(f"[dim]── turn {turn_idx}  |  goal: {goal_action}  |  tg: {tg_short}{ctx_label}[/dim]")

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
                    console.print()

            if args.output_file:
                _write_jsonl_record(
                    file_path=args.output_file,
                    question=question,
                    answer=answer.text,
                    goal_action=goal_action if not args.fast_path else "fast_path",
                    status=status,
                    task_graph_id=task_graph_id or "",
                    run_id=run_id or "",
                    error=error,
                    context_length=context_length,
                    search_query=search_query,
                    rag_enabled=rag_enabled,
                )
        return
    # ── Interactive REPL ────────────────────────────────────────────────────

    session: PromptSession[str] = PromptSession(history=InMemoryHistory())
    previous_task_graph_id: str | None = None
    last_analysis: RunAnalysis | None = None
    turn = 0

    while True:
        try:
            text = session.prompt(
                [("class:prompt", f"[{turn}]> ")],
                style=STYLE,
            )
        except EOFError:
            console.print("\nGoodbye.")
            break

        text = text.strip()
        if not text:
            continue

        if text == "/quit":
            console.print("Goodbye.")
            break

        if text == "/help":
            console.print(
                "Available commands:\n"
                "  /help            — this message\n"
                "  /quit            — exit\n"
                "  /clear           — close active goal and reset conversation\n"
                "  /inspect         — show workspace memory items and edge count\n"
                "  /graph           — show the task graph from the last turn\n"
                "  /run             — show node states from the last run\n"
                "  /export          — export all turns to the default output file (requires --output-file)\n"
                "  /export <path>   — export all turns to the specified JSONL file\n"
                "  /inject <text>   — add a context note before the next turn\n"
                "  anything else is sent to llmasm"
            )
            continue

        if text == "/clear":
            active_goal = goal_tracker.load_active_goal(workspace_id)
            if active_goal:
                goal_tracker.close_goal(active_goal.id, reason="user cleared conversation")
            previous_task_graph_id = None
            console.print("[dim]Conversation cleared. Active goal closed.[/dim]")
            continue

        if text == "/inspect":
            items = storage.list_memory_items(workspace_id)
            edges = storage.list_workspace_edges(workspace_id)
            if items:
                console.print(f"Workspace memory ({len(items)} items):")
                for item in items:
                    snippet = item.text[:60].replace("\n", " ")
                    console.print(
                        f"  [{item.id[:16]}...] {item.kind:<16} \"{snippet}...\"",
                        style="dim",
                    )
            else:
                console.print("[dim]No memory items yet.[/dim]")
            console.print(f"Workspace edges: {len(edges)}", style="dim")
            for edge in edges:
                console.print(
                    f"  {edge.edge_type:<16} {edge.from_id[:16]}... → {edge.to_id[:16]}...",
                    style="dim",
                )
            continue

        if text == "/graph":
            if last_analysis is None:
                console.print("[dim]No task graph yet — run a prompt first.[/dim]")
            else:
                tg = last_analysis.task_graph
                state_by_node = {s.node_id: s for s in last_analysis.node_states}
                has_router = any(n.kind == "router" for n in tg.nodes)
                console.print(
                    f"[bold]Task graph[/bold] {tg.id}  "
                    f"({len(tg.nodes)} nodes, {len(tg.task_edges)} edges)"
                    + ("  [bold yellow][router][/bold yellow]" if has_router else "  [dim][no router][/dim]")
                )
                node_by_id = {n.id: n for n in tg.nodes}
                for node in tg.nodes:
                    state = state_by_node.get(node.id)
                    status = state.status if state else "?"
                    status_colour = {"succeeded": "green", "failed": "red", "skipped": "yellow"}.get(status, "dim")
                    extra = ""
                    if node.kind == "router" and state and state.status == "succeeded":
                        branch = state.metadata.get("selected_branch", "")
                        extra = f"  [bold cyan]→ branch={branch}[/bold cyan]" if branch else ""
                    elif status == "skipped":
                        reason = (state.metadata or {}).get("skip_reason", "")
                        extra = f"  [dim]({reason})[/dim]" if reason else ""
                    console.print(
                        f"  [{status_colour}]{status:<12}[/{status_colour}]  "
                        f"[bold]{node.kind:<14}[/bold]  {node.name}  [dim]{node.id[:20]}[/dim]{extra}"
                    )
                if tg.task_edges:
                    console.print("  [dim]edges:[/dim]")
                    for edge in tg.task_edges:
                        src = node_by_id.get(edge.from_node_id)
                        dst = node_by_id.get(edge.to_node_id)
                        branch = edge.metadata.get("branch", "")
                        dst_state = state_by_node.get(edge.to_node_id)
                        dst_status = dst_state.status if dst_state else "?"
                        branch_label = f"  [dim]branch={branch}[/dim]" if branch else ""
                        pruned = "  [dim](pruned)[/dim]" if dst_status == "skipped" else ""
                        console.print(
                            f"    {(src.name if src else edge.from_node_id[:16])}"
                            f" ──[{edge.from_port}]──▶ [{edge.to_port}]"
                            f" {(dst.name if dst else edge.to_node_id[:16])}{branch_label}{pruned}"
                        )
            continue

        if text == "/run":
            if last_analysis is None:
                console.print("[dim]No run yet — run a prompt first.[/dim]")
            else:
                node_by_id = {n.id: n for n in last_analysis.task_graph.nodes}
                console.print(f"[bold]Run[/bold] {last_analysis.run.id}  status={last_analysis.run.status}")
                for state in last_analysis.node_states:
                    node = node_by_id.get(state.node_id)
                    name = node.name if node else state.node_id[:20]
                    kind = node.kind if node else "?"
                    status_colour = {"succeeded": "green", "failed": "red", "skipped": "yellow"}.get(state.status, "dim")
                    line = (
                        f"  [{status_colour}]{state.status:<12}[/{status_colour}]  "
                        f"[bold]{kind:<14}[/bold]  {name}"
                    )
                    if state.last_error:
                        line += f"  [red]{str(state.last_error)[:80]}[/red]"
                    console.print(line)
                usage = last_analysis.token_usage()
                console.print(
                    f"  [dim]tokens: in={usage['input_tokens']}  out={usage['output_tokens']}  "
                    f"artifacts={usage['artifact_tokens']}[/dim]"
                )
            continue

        if text.startswith("/export "):
            path = text[8:].strip()
            if path:
                _export_workspace_turns(path, workspace_id, storage, console)
            continue

        if text == "/export":
            if not args.output_file:
                console.print("[red]No default output file set. Use /export <path> or start with --output-file.[/red]")
                continue
            _export_workspace_turns(args.output_file, workspace_id, storage, console)
            continue

        if text.startswith("/inject "):
            note = text[8:].strip()
            if note:
                storage.persist_memory_item(
                    MemoryItem(
                        id=new_id("memory"),
                        workspace_graph_id=workspace_id,
                        kind="human_note",
                        text=note,
                        metadata={"injected_before_turn": turn},
                    )
                )
                console.print("[dim]Context injected.[/dim]")
            continue

        console.print()

        turn_status = "succeeded"
        turn_error: str | None = None
        turn_task_graph_id: str | None = None
        turn_run_id: str | None = None
        turn_analysis: RunAnalysis | None = None
        turn_answer = FinalAnswer(text="", sources=[])
        turn_goal_action = ""

        turn_context_length: int | None = None
        turn_search_query: str | None = None
        turn_rag_enabled = False

        if args.fast_path:
            turn_info: dict[str, Any] = {}
            try:
                turn_answer = app.chat(workspace_id, text, out_info=turn_info, turn=turn)
                turn_context_length = turn_info.get("instruction_tokens")
                turn_run_id = turn_info.get("run_id", "")
                turn_search_query = turn_info.get("search_query")
                turn_rag_enabled = bool(turn_info.get("rag_enabled", False))
            except LLMASMError as exc:
                turn_status = "failed"
                turn_error = str(exc)
                console.print(f"[red]Error: {exc}[/red]")
            if turn_answer.text:
                console.print(Markdown(turn_answer.text))
                if turn_answer.sources:
                    console.print(f"[dim]sources: {turn_answer.sources}[/dim]")
            else:
                console.print("[dim](no output)[/dim]")
            ctx_label = f"  |  ctx={turn_context_length}tk" if turn_context_length is not None else ""
            console.print(f"[dim]── turn {turn}  |  [fast path]{ctx_label}[/dim]")
            console.print()
        else:
            try:
                turn_task_graph_id = app.compile(workspace_id, text)
                turn_run_id = app.run(turn_task_graph_id)
                turn_analysis = app.query_run(turn_run_id)
                last_analysis = turn_analysis
                if turn_analysis:
                    prompt_artifacts = [a for a in turn_analysis.artifacts if a.port == "prompt"]
                    turn_context_length = sum(a.token_count for a in prompt_artifacts) if prompt_artifacts else None
            except CompilationError as exc:
                turn_status = "failed"
                turn_error = f"Planner failed to compile after {exc.attempts} attempt(s): {exc.last_errors}"
                console.print(
                    f"[red]Planner failed to compile after {exc.attempts} attempt(s).[/red]"
                )
                if exc.last_errors:
                    console.print(f"[red]Last errors: {exc.last_errors}[/red]")
                previous_task_graph_id = None
            except LLMASMError as exc:
                turn_status = "failed"
                turn_error = str(exc)
                console.print(f"[red]Error: {exc}[/red]")
                previous_task_graph_id = None

            if turn_analysis:
                final_id = _final_node_id(turn_analysis)
                turn_answer = _extract_answer(turn_analysis, final_id)

                if turn_answer.text:
                    console.print(Markdown(turn_answer.text))
                    if turn_answer.sources:
                        console.print(f"[dim]sources: {turn_answer.sources}[/dim]")
                else:
                    console.print("[dim](no output)[/dim]")

                turn_goal_action = turn_analysis.task_graph.metadata.get("goal_action", "")
                tg_short = turn_task_graph_id.split("_")[-1][:12]
                ctx_label = f"  |  ctx={turn_context_length}tk" if turn_context_length is not None else ""
                console.print(f"[dim]── turn {turn}  |  goal: {turn_goal_action}  |  tg: {tg_short}{ctx_label}[/dim]")

                # Persist turn as a MemoryItem so future turns can retrieve it as context.
                # write_memory_item also embeds the text when embeddings_enabled=True.
                memory = write_memory_item(
                    workspace_graph_id=workspace_id,
                    kind="turn",
                    text=f"Q: {text}\nA: {turn_answer.text}",
                    runtime_config=app.runtime_config,
                    provider=app.provider,
                    embedding_store=app.embedding_store,
                    storage=storage,
                    source_run_id=turn_run_id,
                )

                # FOLLOWS_UP: new task graph → previous task graph
                if previous_task_graph_id is not None:
                    storage.persist_workspace_edge(
                        WorkspaceEdge(
                            id=new_id("edge"),
                            workspace_graph_id=workspace_id,
                            edge_type=WorkspaceEdgeType.FOLLOWS_UP,
                            from_type="task_graph",
                            from_id=turn_task_graph_id,
                            to_type="task_graph",
                            to_id=previous_task_graph_id,
                            reason=f"turn {turn} follows turn {turn - 1}",
                        )
                    )

                # PRODUCED: final node → MemoryItem
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

                previous_task_graph_id = turn_task_graph_id
                console.print()


        if args.output_file:
            _write_jsonl_record(
                file_path=args.output_file,
                question=text,
                answer=turn_answer.text,
                goal_action=turn_goal_action if not args.fast_path else "fast_path",
                status=turn_status,
                task_graph_id=turn_task_graph_id or "",
                run_id=turn_run_id or "",
                error=turn_error,
                context_length=turn_context_length,
                search_query=turn_search_query,
                rag_enabled=turn_rag_enabled,
            )

        turn += 1


if __name__ == "__main__":
    main()
