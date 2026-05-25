"""Autonomous graph compilation benchmark over an open QA dataset.

This example downloads SQuAD 2.0 dev data, builds a small lexical retriever, and
asks LLMASM to compile a new graph for each natural-language QA prompt. Unlike
``complex_end_to_end.py``, this does not feed a fixed proposal to the compiler:
the planner model must emit each task graph.

Default run:
    python examples/open_dataset_qa.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmasm.api import LLMASM
from llmasm.analysis.visualize import to_viewer_graph
from llmasm.config import RuntimeConfig
from llmasm.errors import LLMASMError
from llmasm.graph.registry import default_schema_registry
from llmasm.providers.base import EmbeddingOutput, ModelInfo, ModelOutput
from llmasm.providers.ollama import OllamaProvider
from llmasm.schemas import RawText
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.base import ToolSpec
from llmasm.tools.registry import ToolRegistry

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


@dataclass(frozen=True)
class Paragraph:
    """One searchable dataset paragraph."""

    id: str
    title: str
    context: str
    questions: list[dict[str, Any]]


@dataclass(frozen=True)
class BenchmarkCase:
    """One selected benchmark prompt."""

    question: str
    answer: str
    paragraph_id: str
    title: str


class SquadRetrieverTool:
    """Lexical retriever over SQuAD paragraphs."""

    def __init__(self, paragraphs: list[Paragraph], top_k: int = 4) -> None:
        self.paragraphs = paragraphs
        self.top_k = top_k

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="squad.search_passages",
            description=(
                "Search SQuAD 2.0 Wikipedia passages for evidence relevant to a question. "
                "Input is RawText containing the user question. Output is RawText with the "
                "top evidence passages. Use this tool before any model node for dataset QA."
            ),
            input_schema="RawText",
            output_schema="RawText",
        )

    def invoke(self, input: BaseModel) -> BaseModel:
        query = extract_question(getattr(input, "text", ""))
        ranked = rank_paragraphs(query, self.paragraphs)[: self.top_k]
        blocks = [f"question: {query}"]
        for score, paragraph in ranked:
            blocks.append(
                "\n".join(
                    [
                        f"passage_id: {paragraph.id}",
                        f"title: {paragraph.title}",
                        f"score: {score:.3f}",
                        "context:",
                        paragraph.context,
                    ]
                )
            )
        return RawText(text="\n\n---\n\n".join(blocks))


class PromptDumpingProvider:
    """Provider wrapper that records prompts before forwarding calls."""

    name = "prompt-dump"

    def __init__(self, inner: OllamaProvider, dump_dir: Path, available_models: list[str]) -> None:
        self.inner = inner
        self.dump_dir = dump_dir
        self.available_models = available_models
        self.call_index = 0
        self.dump_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[ModelInfo]:
        self._write_json(
            "list-models",
            {
                "base_url": self.inner.base_url,
                "models": self.available_models,
                "source": "configured debug models",
                "timeout": self.inner.timeout,
            },
        )
        print(f"debug_provider_call: list_models using configured models -> {self.dump_dir}")
        return [ModelInfo(name=name) for name in self.available_models]

    def generate(
        self,
        prompt: str,
        options: dict[str, Any] | None = None,
        format_schema: dict[str, Any] | None = None,
    ) -> ModelOutput:
        model = str((options or {}).get("model", self.inner.default_model))
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        stem = self._next_stem(f"generate-{safe_model}")
        prompt_path = self.dump_dir / f"{stem}.prompt.txt"
        meta_path = self.dump_dir / f"{stem}.meta.json"
        prompt_path.write_text(prompt, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "options": options or {},
                    "format_schema": format_schema,
                    "prompt_chars": len(prompt),
                    "prompt_words": len(prompt.split()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"debug_prompt: {prompt_path}")
        try:
            output = self.inner.generate(prompt, options, format_schema)
        except Exception as exc:
            error_path = self.dump_dir / f"{stem}.error.txt"
            error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            raise
        response_path = self.dump_dir / f"{stem}.response.txt"
        raw_path = self.dump_dir / f"{stem}.response.json"
        response_path.write_text(output.text, encoding="utf-8")
        raw_path.write_text(json.dumps(output.raw or {}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output

    def embed(
        self,
        texts: list[str],
        options: dict[str, Any] | None = None,
    ) -> list[EmbeddingOutput]:
        self._write_json(
            "embed",
            {
                "options": options or {},
                "text_count": len(texts),
                "text_chars": [len(text) for text in texts],
            },
        )
        return self.inner.embed(texts, options)

    def _write_json(self, label: str, payload: dict[str, Any]) -> None:
        path = self.dump_dir / f"{self._next_stem(label)}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _next_stem(self, label: str) -> str:
        self.call_index += 1
        return f"{self.call_index:03d}-{label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="data/squad/dev-v2.0.json")
    parser.add_argument("--download-url", default=SQUAD_DEV_URL)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Select cases and verify retriever hits without calling Ollama.",
    )
    parser.add_argument("--max-paragraphs", type=int, default=450)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--planner-model", default="gemma4:26b")
    parser.add_argument("--runtime-model", default="gemma4:e4b")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--compiler-attempts", type=int, default=3)
    parser.add_argument("--report-jsonl", default="data/squad/llmasm_qa_report.jsonl")
    parser.add_argument(
        "--summarize-report",
        default=None,
        help="Summarize an existing JSONL report and exit.",
    )
    parser.add_argument("--write-viewer-json", default=None)
    parser.add_argument(
        "--debug-prompts",
        action="store_true",
        help="Dump planner/runtime prompts and provider call metadata to disk.",
    )
    parser.add_argument("--debug-prompts-dir", default="data/squad/debug_prompts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize_report:
        summarize_report(Path(args.summarize_report))
        return
    dataset_path = Path(args.dataset_path)
    ensure_dataset(dataset_path, args.download_url)
    if args.download_only:
        print(f"dataset_path: {dataset_path}")
        return

    paragraphs = load_squad_paragraphs(dataset_path, args.max_paragraphs)
    cases = select_cases(paragraphs, args.limit, args.top_k)
    if len(cases) < args.limit:
        raise SystemExit(f"Only found {len(cases)} retrievable cases")
    if args.retrieval_only:
        retriever = SquadRetrieverTool(paragraphs, top_k=args.top_k)
        for index, case in enumerate(cases, start=1):
            output = retriever.invoke(RawText(text=case.question))
            hit = contains_answer(output.text, case.answer)
            print(f"[{index}] retrieval_hit={hit} title={case.title}")
            print(f"question: {case.question}")
            print(f"gold: {case.answer}")
        return

    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    tools.register(SquadRetrieverTool(paragraphs, top_k=args.top_k))
    provider = OllamaProvider(
        base_url=args.ollama_url,
        timeout=args.timeout,
        default_model=args.runtime_model,
    )
    if args.debug_prompts:
        provider = PromptDumpingProvider(
            provider,
            Path(args.debug_prompts_dir),
            sorted({args.planner_model, args.runtime_model}),
        )
    storage = InMemoryStorage()
    app = LLMASM(
        storage=storage,
        provider=provider,
        tool_registry=tools,
        schema_registry=schemas,
        runtime_config=RuntimeConfig(
            planner_model=args.planner_model,
            default_model=args.runtime_model,
            compiler_max_attempts=args.compiler_attempts,
            planner_max_tokens=8192,
        ),
    )

    report_path = Path(args.report_jsonl)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    last_analysis = None
    with report_path.open("w", encoding="utf-8") as report:
        for index, case in enumerate(cases, start=1):
            result = run_case(app, case, index)
            last_analysis = result.pop("analysis", None)
            report.write(json.dumps(result, sort_keys=True) + "\n")
            report.flush()
            print_result(result)

    if args.write_viewer_json and last_analysis is not None:
        target = Path(args.write_viewer_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(to_viewer_graph(last_analysis), indent=2) + "\n", encoding="utf-8")
        print(f"viewer_json: {target}")
    print(f"report_jsonl: {report_path}")
    summarize_report(report_path)


def ensure_dataset(path: Path, url: str) -> None:
    """Download the dataset if it is not present."""

    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SQuAD 2.0 dev data from {url}")
    urlretrieve(url, path)


def load_squad_paragraphs(path: Path, max_paragraphs: int) -> list[Paragraph]:
    data = json.loads(path.read_text(encoding="utf-8"))
    paragraphs: list[Paragraph] = []
    for article in data["data"]:
        title = article["title"]
        for item in article["paragraphs"]:
            paragraph = Paragraph(
                id=f"p{len(paragraphs):05d}",
                title=title,
                context=normalize_space(item["context"]),
                questions=item["qas"],
            )
            paragraphs.append(paragraph)
            if len(paragraphs) >= max_paragraphs:
                return paragraphs
    return paragraphs


def select_cases(
    paragraphs: list[Paragraph],
    limit: int,
    top_k: int,
) -> list[BenchmarkCase]:
    """Select answerable questions whose answer appears in retrieved passages."""

    cases: list[BenchmarkCase] = []
    for paragraph in paragraphs:
        for qa in paragraph.questions:
            if qa.get("is_impossible"):
                continue
            answers = qa.get("answers") or []
            if not answers:
                continue
            answer = normalize_space(answers[0]["text"])
            if len(answer) < 3:
                continue
            question = normalize_space(qa["question"])
            ranked = rank_paragraphs(question, paragraphs)[:top_k]
            retrieved_text = "\n".join(item.context for _, item in ranked)
            if contains_answer(retrieved_text, answer):
                cases.append(
                    BenchmarkCase(
                        question=question,
                        answer=answer,
                        paragraph_id=paragraph.id,
                        title=paragraph.title,
                    )
                )
                if len(cases) >= limit:
                    return cases
    return cases


def run_case(app: LLMASM, case: BenchmarkCase, index: int) -> dict[str, Any]:
    prompt = (
        "Answer this dataset question using the registered SQuAD passage search tool. "
        "Use retrieved evidence, answer concisely, and do not use outside knowledge.\n\n"
        f"Question: {case.question}"
    )
    workspace_id = app.create_workspace(f"squad-eval-{index}")
    started = time.perf_counter()
    try:
        task_graph_id = app.compile(workspace_id, prompt)
        run_id = app.run(task_graph_id)
        analysis = app.query_run(run_id)
        answer = final_answer_text(analysis)
        retrieved = retrieved_text(analysis)
        return {
            "case": index,
            "ok": True,
            "question": case.question,
            "gold_answer": case.answer,
            "title": case.title,
            "task_graph_id": task_graph_id,
            "run_id": run_id,
            "compiled_nodes": len(analysis.task_graph.nodes),
            "tool_calls": len(analysis.tool_calls),
            "model_calls": len(analysis.model_calls),
            "retrieval_hit": contains_answer(retrieved, case.answer),
            "answer_contains_gold": contains_answer(answer, case.answer),
            "answer": answer,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "analysis": analysis,
        }
    except LLMASMError as exc:
        return {
            "case": index,
            "ok": False,
            "question": case.question,
            "gold_answer": case.answer,
            "title": case.title,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "last_errors": str(getattr(exc, "last_errors", "")),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }


def final_answer_text(analysis: Any) -> str:
    final_node_ids = {node.id for node in analysis.task_graph.nodes if node.kind == "final"}
    artifacts = [artifact for artifact in analysis.artifacts if artifact.node_id in final_node_ids]
    if not artifacts:
        return ""
    content = artifacts[-1].content_json
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content)


def retrieved_text(analysis: Any) -> str:
    tool_node_ids = {call.node_id for call in analysis.tool_calls}
    chunks = []
    for artifact in analysis.artifacts:
        if artifact.node_id in tool_node_ids:
            content = artifact.content_json
            if isinstance(content, dict):
                chunks.append(str(content.get("text", "")))
            else:
                chunks.append(str(content))
    return "\n".join(chunks)


def print_result(result: dict[str, Any]) -> None:
    status = "ok" if result["ok"] else "failed"
    print(f"\n[{result['case']}] {status}: {result['question']}")
    print(f"gold: {result['gold_answer']}")
    if result["ok"]:
        print(f"retrieval_hit: {result['retrieval_hit']}")
        print(f"answer_contains_gold: {result['answer_contains_gold']}")
        print(f"nodes/tool/model: {result['compiled_nodes']}/{result['tool_calls']}/{result['model_calls']}")
        print(f"answer: {result['answer'][:500]}")
    else:
        print(f"{result['error_type']}: {result['error']}")


def summarize_report(path: Path) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        print(f"summary: no rows in {path}")
        return
    ok_rows = [row for row in rows if row.get("ok")]
    retrieval_hits = [row for row in ok_rows if row.get("retrieval_hit")]
    answer_hits = [row for row in ok_rows if row.get("answer_contains_gold")]
    failure_counts: dict[str, int] = {}
    for row in rows:
        if row.get("ok"):
            continue
        key = str(row.get("error_type") or "unknown")
        failure_counts[key] = failure_counts.get(key, 0) + 1
    elapsed = [float(row.get("elapsed_seconds") or 0) for row in rows]
    print("\nsummary")
    print(f"cases: {len(rows)}")
    print(f"ok: {len(ok_rows)}/{len(rows)}")
    print(f"retrieval_hit: {len(retrieval_hits)}/{len(ok_rows)}")
    print(f"answer_contains_gold: {len(answer_hits)}/{len(ok_rows)}")
    print(f"avg_elapsed_seconds: {sum(elapsed) / len(elapsed):.2f}")
    if failure_counts:
        print("failures: " + ", ".join(f"{key}={value}" for key, value in sorted(failure_counts.items())))


def rank_paragraphs(query: str, paragraphs: list[Paragraph]) -> list[tuple[float, Paragraph]]:
    query_terms = weighted_terms(query)
    ranked = []
    for paragraph in paragraphs:
        text_terms = weighted_terms(paragraph.title + " " + paragraph.context)
        if not query_terms or not text_terms:
            score = 0.0
        else:
            overlap = set(query_terms) & set(text_terms)
            score = sum(query_terms[term] for term in overlap) / sum(query_terms.values())
        ranked.append((score, paragraph))
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def weighted_terms(text: str) -> dict[str, float]:
    terms: dict[str, float] = {}
    for token in re.findall(r"[A-Za-z0-9]+", text.lower()):
        if token in STOPWORDS or len(token) < 3:
            continue
        terms[token] = terms.get(token, 0.0) + 1.0
    return terms


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_question(text: str) -> str:
    match = re.search(r"(?is)\bquestion\s*:\s*(.+)", text)
    if match:
        return normalize_space(match.group(1))
    return normalize_space(text)


def contains_answer(text: str, answer: str) -> bool:
    return normalize_for_match(answer) in normalize_for_match(text)


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "was",
    "were",
    "are",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "did",
    "does",
    "has",
    "have",
    "had",
    "into",
    "about",
    "using",
    "use",
    "not",
    "its",
    "his",
    "her",
    "their",
    "they",
    "them",
}


if __name__ == "__main__":
    main()
