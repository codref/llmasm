"""Generate a questions.txt file from an AMI Meeting Corpus transcript.

This script picks one meeting transcript (or reads a custom transcript file),
then writes a questions.txt with the full transcript as a context-setting
first turn followed by comprehension questions — one question per line — so
``chat.py --input-file`` can answer them turn by turn.

Data source: AMI Meeting Corpus (open source, ~100 hours of real meetings
recorded at the University of Edinburgh).

Dependencies (opt-in, install with ``pip install llmasm[ami]``):
  datasets>=2.14   – automatic AMI download via HuggingFace

Usage::

    # List available meetings
    python examples/generate_ami_questions.py --list-meetings

    # Pick the first meeting, default questions
    python examples/generate_ami_questions.py --output questions.txt

    # Pick a specific meeting, generate 5 questions via the LLM
    python examples/generate_ami_questions.py --meeting-index 3 \\
        --generate-questions --max-questions 5 --output questions.txt

    # Bring your own transcript (e.g. a real call transcript) — no AMI required
    python examples/generate_ami_questions.py \\
        --transcript-file my_call.txt --output questions.txt

    # Then feed it to the chat tool
    python examples/chat.py --input-file questions.txt
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmasm.errors import ProviderError
from llmasm.providers.ollama import OllamaProvider

DEFAULT_OUTPUT = Path("questions.txt")
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ami"

# Maximum transcript characters included in the context-setting prompt.
# AMI meetings are 30–60 min and transcripts can exceed 10k words; truncate
# to keep the first turn manageable (mirrors CoQA's MAX_PASSAGE_CHARS).
MAX_TRANSCRIPT_CHARS = 4000

# Generic meeting-comprehension questions used in template mode. They are
# intentionally broad so they work on any AMI meeting; --generate-questions
# produces meeting-specific alternatives.
TEMPLATE_QUESTIONS: tuple[str, ...] = (
    "What was the main topic or agenda of this meeting?",
    "What decisions were made during the meeting?",
    "Who led or facilitated the meeting?",
    "What action items or tasks were assigned, and to whom?",
    "Were there any disagreements or differing opinions discussed?",
    "What was the overall conclusion or outcome?",
    "Did the participants reach a consensus on any issue?",
    "What data, evidence, or examples were presented?",
    "Were there any deadlines, dates, or timelines mentioned?",
    "What was the general mood or tone of the discussion?",
    "What problems or challenges were raised?",
    "What proposals or ideas were suggested?",
    "Which participant spoke the most, and what did they focus on?",
    "What open questions remained unresolved at the end?",
    "If a follow-up meeting was planned, what was the next step?",
)

QUESTION_GEN_PROMPT = (
    "You generate comprehension questions about meeting transcripts.\n"
    "Read the transcript below and produce exactly {n} distinct questions "
    "that test understanding of its content (decisions, action items, "
    "participants' positions, outcomes, etc.).\n"
    "Return ONLY the questions, one per line, with no numbering, bullets, "
    "or preamble.\n\n"
    "Transcript:\n{transcript}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a questions.txt file from an AMI meeting transcript"
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        metavar="PATH",
        help=f"Directory to cache AMI data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--transcript-file",
        default=None,
        metavar="PATH",
        help="Use a local transcript (txt or srt) instead of downloading AMI",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        metavar="PATH",
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--meeting-index",
        type=int,
        default=None,
        metavar="N",
        help="0-based index of the meeting to use (default: first meeting)",
    )
    parser.add_argument(
        "--list-meetings",
        action="store_true",
        default=False,
        help="Print a summary of available meetings and exit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of meetings to show with --list-meetings (default: 20)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of questions to include (default: 5)",
    )
    parser.add_argument(
        "--max-transcript-chars",
        type=int,
        default=MAX_TRANSCRIPT_CHARS,
        metavar="N",
        help=f"Truncate transcript to N chars in the context prompt "
        f"(default: {MAX_TRANSCRIPT_CHARS})",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        default=False,
        help="Omit the transcript context prompt (questions only)",
    )
    parser.add_argument(
        "--generate-questions",
        action="store_true",
        default=False,
        help="Use an LLM to generate meeting-specific questions instead of templates",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--question-model",
        default="llama3.1:8b",
        metavar="MODEL",
        help="Ollama model used for --generate-questions (default: llama3.1:8b)",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for template question selection",
    )
    return parser.parse_args()


def normalize_transcript(text: str) -> str:
    """Collapse all whitespace to single spaces and strip ends.

    ``chat.py`` splits the input file by newlines and treats each line as a
    turn, so the transcript must be a single line to avoid being parsed as
    many short turns.
    """
    return re.sub(r"\s+", " ", text).strip()


def truncate_transcript(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + " [...]"
    return text


def load_transcript_from_file(path: Path) -> tuple[str, str]:
    """Return (meeting_id, transcript_text) from a local file."""
    text = path.read_text(encoding="utf-8")
    return path.stem, normalize_transcript(text)


def load_ami_meetings(data_dir: Path) -> list[tuple[str, str]]:
    """Download AMI via HuggingFace ``datasets`` and return [(meeting_id, transcript)].

    Caches the download under ``data_dir``. Each meeting's transcript is
    normalised to a single-line string of speaker turns.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' package is required to download AMI automatically.\n"
            "Install it with:  pip install 'llmasm[ami]'\n"
            "Or use --transcript-file to feed a local transcript instead."
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading AMI corpus (cached in {data_dir}) …", flush=True)
    dataset = load_dataset("ami", "default", cache_dir=str(data_dir))

    # AMI exposes 'train' / 'validation' / 'test'. Concatenate and keep the
    # speaker-annotated transcript text per meeting.
    meetings: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for split in dataset.values():
        for example in split:
            meeting_id = example.get("meeting_id") or ""
            if not meeting_id or meeting_id in seen_ids:
                continue
            seen_ids.add(meeting_id)
            text = example.get("text") or ""
            if not text:
                words = example.get("words") or []
                text = " ".join(str(w) for w in words)
            if text:
                meetings.append((meeting_id, normalize_transcript(text)))
    return meetings


def list_meetings(meetings: list[tuple[str, str]], limit: int) -> None:
    print(f"Total meetings: {len(meetings)}")
    for i, (mid, text) in enumerate(meetings[:limit]):
        snippet = text[:80].replace("\n", " ")
        print(f"[{i:4d}]  {mid}  {len(text):6d} chars  {snippet!r}")
    if len(meetings) > limit:
        print(f"  … {len(meetings) - limit} more (use --limit to show more)")


def pick_meeting(
    meetings: list[tuple[str, str]],
    index: int | None,
    seed: int | None,
) -> tuple[str, str]:
    if not meetings:
        print("No meetings available.", file=sys.stderr)
        sys.exit(1)
    if index is not None:
        if index < 0 or index >= len(meetings):
            print(
                f"--meeting-index {index} is out of range (0–{len(meetings) - 1}).",
                file=sys.stderr,
            )
            sys.exit(1)
        return meetings[index]
    if seed is not None:
        random.seed(seed)
        return random.choice(meetings)
    return meetings[0]


def pick_template_questions(n: int, seed: int | None) -> list[str]:
    pool = list(TEMPLATE_QUESTIONS)
    if seed is not None:
        random.seed(seed)
    random.shuffle(pool)
    return pool[:n]


def generate_llm_questions(
    transcript: str,
    n: int,
    ollama_url: str,
    model: str,
    timeout: float,
) -> list[str]:
    provider = OllamaProvider(base_url=ollama_url, timeout=timeout, default_model=model)
    prompt = QUESTION_GEN_PROMPT.format(n=n, transcript=transcript)
    try:
        output = provider.generate(prompt, options={"model": model})
    except ProviderError as exc:
        print(f"LLM question generation failed: {exc}", file=sys.stderr)
        sys.exit(1)
    lines = [line.strip(" -\t") for line in output.text.splitlines()]
    questions = [line for line in lines if line.endswith("?") and len(line) > 5]
    if not questions:
        # Fallback: accept any non-empty line the model emitted.
        questions = [line for line in lines if line]
    return questions[:n]


def build_lines(
    transcript: str,
    questions: list[str],
    max_transcript_chars: int,
    no_transcript: bool,
) -> list[str]:
    lines: list[str] = []
    if not no_transcript:
        body = truncate_transcript(transcript, max_transcript_chars)
        lines.append(
            "I will ask you several questions about the following meeting transcript. "
            "Please read it carefully and answer based on its content.\n\n" + body
        )
    lines.append("")  # visual separator — chat.py skips empty lines
    lines.extend(questions)
    return lines


def main() -> None:
    args = parse_args()

    if args.transcript_file:
        meeting_id, transcript = load_transcript_from_file(Path(args.transcript_file))
        meetings: list[tuple[str, str]] = [(meeting_id, transcript)]
    else:
        meetings = load_ami_meetings(Path(args.data_dir))

    if args.list_meetings:
        list_meetings(meetings, args.limit)
        return

    meeting_id, transcript = pick_meeting(meetings, args.meeting_index, args.seed)
    print(
        f"Selected meeting: id={meeting_id}  "
        f"transcript_chars={len(transcript)}"
    )

    if args.generate_questions:
        questions = generate_llm_questions(
            transcript=truncate_transcript(transcript, args.max_transcript_chars),
            n=args.max_questions,
            ollama_url=args.ollama_url,
            model=args.question_model,
            timeout=args.timeout,
        )
    else:
        questions = pick_template_questions(args.max_questions, args.seed)

    if not questions:
        print("No questions produced; aborting.", file=sys.stderr)
        sys.exit(1)

    lines = build_lines(
        transcript=transcript,
        questions=questions,
        max_transcript_chars=args.max_transcript_chars,
        no_transcript=args.no_transcript,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    n_turns = sum(1 for ln in lines if ln.strip())
    print(f"Wrote {n_turns} turn(s) ({len(questions)} questions) to {output}")
    print(f"\nRun with:\n  python examples/chat.py --input-file {output}")


if __name__ == "__main__":
    main()
