"""Generate a questions.txt file from a CoQA story for use with chat.py --input-file.

CoQA (Conversational Question Answering) contains stories (passages) with
multi-turn Q&A dialogues. This script picks one story and writes its questions
to a plain-text file (one question per line), prepending a context-setting
prompt so the LLM has the passage to reason over.

Usage:
    # Pick the first story (default)
    python examples/generate_questions.py --output questions.txt

    # Pick story by 0-based index
    python examples/generate_questions.py --story-index 3 --output questions.txt

    # Pick a random story from a specific domain
    python examples/generate_questions.py --domain wikipedia --output questions.txt

    # List available stories without generating output
    python examples/generate_questions.py --list-stories --limit 20

    # Then pass the file to the chat tool
    python examples/chat.py --input-file questions.txt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from urllib.request import urlretrieve

COQA_DEV_URL = "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "coqa"
DEFAULT_OUTPUT = Path("questions.txt")

# Maximum passage characters included in the context-setting prompt.
# CoQA passages can be very long; we truncate to keep the first turn manageable.
MAX_PASSAGE_CHARS = 3000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a questions.txt file from a CoQA story"
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        metavar="PATH",
        help=f"Directory to cache the CoQA dev file (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--coqa-url",
        default=COQA_DEV_URL,
        metavar="URL",
        help="URL of the CoQA dev JSON file",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        metavar="PATH",
        help="Output file path (default: questions.txt)",
    )
    parser.add_argument(
        "--story-index",
        type=int,
        default=None,
        metavar="N",
        help="0-based index of the story to use (default: first story, or random if --domain is set)",
    )
    parser.add_argument(
        "--domain",
        default=None,
        metavar="DOMAIN",
        help="Filter by source domain: wikipedia, reddit, cnn, gutenberg, mctest, science",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of questions to include (default: all)",
    )
    parser.add_argument(
        "--max-passage-chars",
        type=int,
        default=MAX_PASSAGE_CHARS,
        metavar="N",
        help=f"Truncate passage to this many characters in the context prompt (default: {MAX_PASSAGE_CHARS})",
    )
    parser.add_argument(
        "--no-passage",
        action="store_true",
        default=False,
        help="Omit the passage context prompt (questions only)",
    )
    parser.add_argument(
        "--list-stories",
        action="store_true",
        default=False,
        help="Print a summary of available stories and exit (use --limit to cap output)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of stories to show with --list-stories (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for story selection when --domain is set without --story-index",
    )
    return parser.parse_args()


def download_coqa(url: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "coqa-dev-v1.0.json"
    if not dest.exists():
        print(f"Downloading CoQA dev set from {url} …", flush=True)
        urlretrieve(url, dest)
        print(f"Saved to {dest}")
    else:
        print(f"Using cached {dest}")
    return dest


def load_stories(path: Path) -> list[dict]:
    with path.open() as f:
        payload = json.load(f)
    return payload["data"]


def filter_stories(stories: list[dict], domain: str | None) -> list[dict]:
    if domain is None:
        return stories
    return [s for s in stories if s.get("source", "").lower() == domain.lower()]


def pick_story(stories: list[dict], index: int | None, seed: int | None) -> dict:
    if not stories:
        print("No stories match the given filters.", file=sys.stderr)
        sys.exit(1)
    if index is not None:
        if index < 0 or index >= len(stories):
            print(
                f"--story-index {index} is out of range (0–{len(stories) - 1}).",
                file=sys.stderr,
            )
            sys.exit(1)
        return stories[index]
    # Random pick when domain filter is used, deterministic otherwise.
    if seed is not None:
        random.seed(seed)
        return random.choice(stories)
    return stories[0]


def build_lines(
    story: dict,
    max_questions: int | None,
    max_passage_chars: int,
    no_passage: bool,
) -> list[str]:
    lines: list[str] = []

    if not no_passage:
        passage = story.get("story", "")
        if len(passage) > max_passage_chars:
            passage = passage[:max_passage_chars].rstrip() + " [...]"
        lines.append(
            f"I will ask you several questions about the following passage. "
            f"Please read it carefully and answer based on its content.\n\n{passage}"
        )

    questions: list[dict] = story.get("questions", [])
    if max_questions is not None:
        questions = questions[:max_questions]

    for q in questions:
        text = q.get("input_text", "").strip()
        if text:
            lines.append(text)

    return lines


def list_stories(stories: list[dict], limit: int) -> None:
    domains: dict[str, int] = {}
    for s in stories:
        d = s.get("source", "unknown")
        domains[d] = domains.get(d, 0) + 1
    print(f"Total stories: {len(stories)}")
    print(f"Domains: {', '.join(f'{d}={n}' for d, n in sorted(domains.items()))}")
    print()
    for i, s in enumerate(stories[:limit]):
        n_q = len(s.get("questions", []))
        passage_snippet = s.get("story", "")[:80].replace("\n", " ")
        print(f"[{i:4d}]  {s.get('source', '?'):<12}  {n_q:2d} turns  {passage_snippet!r}")
    if len(stories) > limit:
        print(f"  … {len(stories) - limit} more (use --limit to show more)")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    coqa_path = download_coqa(args.coqa_url, data_dir)
    stories = load_stories(coqa_path)
    filtered = filter_stories(stories, args.domain)

    if args.list_stories:
        list_stories(filtered if args.domain else stories, args.limit)
        return

    story = pick_story(filtered, args.story_index, args.seed)
    n_q = len(story.get("questions", []))
    source = story.get("source", "?")
    story_id = story.get("id", "?")
    print(
        f"Selected story: id={story_id}  source={source}  turns={n_q}  "
        f"passage_chars={len(story.get('story', ''))}"
    )

    lines = build_lines(
        story,
        max_questions=args.max_questions,
        max_passage_chars=args.max_passage_chars,
        no_passage=args.no_passage,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    n_turns = len(lines)
    print(f"Wrote {n_turns} line(s) to {output}")
    print(f"\nRun with:\n  python examples/chat.py --input-file {output}")


if __name__ == "__main__":
    main()
