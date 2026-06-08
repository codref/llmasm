"""Minimal comparison script using Pydantic AI + Ollama.

Mirrors the CLI surface of examples/chat.py (file-driven and REPL modes) but
uses the Pydantic AI agent framework instead of llmasm.  No custom RAG or
embeddings — only the framework-native conversation history.

Usage (file-driven):
    python examples/compare_pydantic_ai.py \
        --model gemma4:e4b-it-qat \
        --input-file questions4.txt \
        --output-file session_pydantic_ai.jsonl

Usage (REPL):
    python examples/compare_pydantic_ai.py --model gemma4:e4b-it-qat
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# prompt-toolkit and rich are optional — gracefully degrade if missing.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.styles import Style
    _HAS_PROMPT_TOOLKIT = True
except Exception:
    _HAS_PROMPT_TOOLKIT = False

try:
    from rich.console import Console
    from rich.markdown import Markdown
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False


STYLE = (
    {"prompt": "#00aa00 bold", "separator": "dim"}
    if _HAS_PROMPT_TOOLKIT
    else {}
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pydantic AI chat comparison")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--timeout", type=float, default=180.0)
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
        help="Path to append JSONL conversation log.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant. Answer the user's question based on the conversation context. If no relevant context is available, answer from your own knowledge.",
        help="System prompt / instruction for the model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = deterministic).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate per response.",
    )
    return parser.parse_args()


def _build_agent(args: argparse.Namespace) -> Agent:
    provider = OllamaProvider(base_url=f"{args.ollama_url.rstrip('/')}/v1")
    model = OllamaModel(args.model, provider=provider)
    settings: dict[str, Any] = {"temperature": args.temperature}
    if args.max_tokens is not None:
        settings["max_tokens"] = args.max_tokens
    return Agent(
        model=model,
        instructions=args.system_prompt,
        model_settings=settings,
    )


def _write_jsonl_record(
    file_path: str,
    question: str,
    answer: str,
    status: str,
    run_id: str,
    error: str | None,
    context_length: int | None = None,
    rag_enabled: bool = False,
) -> None:
    record: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "goal_action": "pydantic_ai",
        "status": status,
        "task_graph_id": "",
        "run_id": run_id,
        "error": error,
        "rag_enabled": rag_enabled,
    }
    if context_length is not None:
        record["context_length"] = context_length
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _run_file_mode(agent: Agent, args: argparse.Namespace) -> None:
    questions = Path(args.input_file).read_text().splitlines()
    questions = [q.strip() for q in questions if q.strip()]

    message_history: list[Any] = []
    console = Console() if _HAS_RICH else None

    for turn_idx, question in enumerate(questions):
        if console:
            console.print(f"[dim][{turn_idx}]>[/dim] {question}")
            console.print()
        else:
            print(f"[{turn_idx}]> {question}")

        status = "succeeded"
        error: str | None = None
        answer_text = ""
        run_id = ""
        context_length: int | None = None

        try:
            result = await agent.run(
                question,
                message_history=message_history,
            )
            answer_text = result.response.text if hasattr(result.response, "text") else str(result.response)
            run_id = result.run_id or str(uuid.uuid4())
            if result.usage:
                context_length = result.usage.total_tokens
            # Append the turn to the conversation history for the next turn
            message_history = result.all_messages()
        except Exception as exc:
            status = "failed"
            error = str(exc)
            if console:
                console.print(f"[red]Error: {exc}[/red]")
            else:
                print(f"Error: {exc}")

        if answer_text:
            if console:
                console.print(Markdown(answer_text))
            else:
                print(answer_text)
        else:
            if console:
                console.print("[dim](no output)[/dim]")
            else:
                print("(no output)")

        ctx_label = f"  |  ctx={context_length}tk" if context_length is not None else ""
        if console:
            console.print(f"[dim]── turn {turn_idx}  |  [pydantic-ai]{ctx_label}[/dim]")
            console.print()
        else:
            print(f"── turn {turn_idx}  |  [pydantic-ai]{ctx_label}")
            print()

        if args.output_file:
            _write_jsonl_record(
                file_path=args.output_file,
                question=question,
                answer=answer_text,
                status=status,
                run_id=run_id,
                error=error,
                context_length=context_length,
                rag_enabled=False,
            )


async def _run_repl_mode(agent: Agent, args: argparse.Namespace) -> None:
    console = Console() if _HAS_RICH else None

    log_label = f"[logging: on → `{args.output_file}`]" if args.output_file else "[logging: off]"
    header = (
        f"**Pydantic AI chat**  |  model: `{args.model}`  |  "
        f"Ollama: `{args.ollama_url}`  |  {log_label}\n"
        f"Type /quit or Ctrl-D to exit."
    )
    if console:
        console.print(Markdown(header))
        console.print("─" * console.width, style="dim")
    else:
        print(header)
        print("-" * 60)

    message_history: list[Any] = []
    turn = 0

    if _HAS_PROMPT_TOOLKIT:
        session: PromptSession[str] = PromptSession(history=InMemoryHistory())
    else:
        session = None

    while True:
        try:
            if session:
                text = session.prompt(
                    [("class:prompt", f"[{turn}]> ")],
                    style=STYLE,
                )
            else:
                text = input(f"[{turn}]> ")
        except (EOFError, KeyboardInterrupt):
            if console:
                console.print("\nGoodbye.")
            else:
                print("\nGoodbye.")
            break

        text = text.strip()
        if not text:
            continue
        if text == "/quit":
            if console:
                console.print("Goodbye.")
            else:
                print("Goodbye.")
            break

        if console:
            console.print()
        else:
            print()

        status = "succeeded"
        error: str | None = None
        answer_text = ""
        run_id = ""
        context_length: int | None = None

        try:
            result = await agent.run(
                text,
                message_history=message_history,
            )
            answer_text = result.response.text if hasattr(result.response, "text") else str(result.response)
            run_id = result.run_id or str(uuid.uuid4())
            if result.usage:
                context_length = result.usage.total_tokens
            message_history = result.all_messages()
        except Exception as exc:
            status = "failed"
            error = str(exc)
            if console:
                console.print(f"[red]Error: {exc}[/red]")
            else:
                print(f"Error: {exc}")

        if answer_text:
            if console:
                console.print(Markdown(answer_text))
            else:
                print(answer_text)
        else:
            if console:
                console.print("[dim](no output)[/dim]")
            else:
                print("(no output)")

        ctx_label = f"  |  ctx={context_length}tk" if context_length is not None else ""
        if console:
            console.print(f"[dim]── turn {turn}  |  [pydantic-ai]{ctx_label}[/dim]")
            console.print()
        else:
            print(f"── turn {turn}  |  [pydantic-ai]{ctx_label}")
            print()

        if args.output_file:
            _write_jsonl_record(
                file_path=args.output_file,
                question=text,
                answer=answer_text,
                status=status,
                run_id=run_id,
                error=error,
                context_length=context_length,
                rag_enabled=False,
            )

        turn += 1


async def _main_async() -> None:
    args = _parse_args()
    agent = _build_agent(args)

    if args.input_file is not None:
        await _run_file_mode(agent, args)
    else:
        await _run_repl_mode(agent, args)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
