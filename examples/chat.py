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
    return parser.parse_args()


def _build_storage(args: argparse.Namespace) -> InMemoryStorage | PostgresStorage:
    if args.db_url:
        return PostgresStorage(args.db_url)
    return InMemoryStorage()


def _build_embedding_store(
    args: argparse.Namespace,
    storage: InMemoryStorage | PostgresStorage,
) -> InMemoryEmbeddingStore | PostgresEmbeddingStore | NullEmbeddingStore:
    if not args.embeddings:
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
        ),
        schema_registry=default_schema_registry(),
        embedding_store=embedding_store,
    )


def main() -> None:
    args = parse_args()
    console = Console()

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

    console.print(
        Markdown(
            f"**llmasm chat**  |  workspace: `{args.workspace_name}`  "
            f"|  model: `{args.runtime_model}`  |  planner: `{args.planner_model}`  "
            f"|  storage: `{storage_label}`  |  {embedding_label}\n"
            f"Type /help for commands, /quit or Ctrl-D to exit."
        )
    )
    console.print("─" * console.width, style="dim")

    session: PromptSession[str] = PromptSession(history=InMemoryHistory())
    previous_task_graph_id: str | None = None
    previous_final_node_id: str | None = None
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
                "  /inject <text>   — add a context note before the next turn\n"
                "  anything else is sent to llmasm"
            )
            continue

        if text == "/clear":
            active_goal = goal_tracker.load_active_goal(workspace_id)
            if active_goal:
                goal_tracker.close_goal(active_goal.id, reason="user cleared conversation")
            previous_task_graph_id = None
            previous_final_node_id = None
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
        try:
            task_graph_id = app.compile(workspace_id, text)
            run_id = app.run(task_graph_id)
            analysis = app.query_run(run_id)
            last_analysis = analysis
        except CompilationError as exc:
            console.print(
                f"[red]Planner failed to compile after {exc.attempts} attempt(s).[/red]"
            )
            if exc.last_errors:
                console.print(f"[red]Last errors: {exc.last_errors}[/red]")
            continue
        except LLMASMError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            continue

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
        console.print(f"[dim]── turn {turn}  |  goal: {goal_action}  |  tg: {tg_short}[/dim]")

        # Persist turn as a MemoryItem so future turns can retrieve it as context.
        # write_memory_item also embeds the text when embeddings_enabled=True.
        memory = write_memory_item(
            workspace_graph_id=workspace_id,
            kind="turn",
            text=f"Q: {text}\nA: {answer.text}",
            runtime_config=app.runtime_config,
            provider=app.provider,
            embedding_store=app.embedding_store,
            storage=storage,
            source_run_id=run_id,
        )

        # FOLLOWS_UP: new task graph → previous task graph
        if previous_task_graph_id is not None:
            storage.persist_workspace_edge(
                WorkspaceEdge(
                    id=new_id("edge"),
                    workspace_graph_id=workspace_id,
                    edge_type=WorkspaceEdgeType.FOLLOWS_UP,
                    from_type="task_graph",
                    from_id=task_graph_id,
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

        previous_task_graph_id = task_graph_id
        previous_final_node_id = final_id
        console.print()
        turn += 1


if __name__ == "__main__":
    main()
