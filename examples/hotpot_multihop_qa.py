"""Autonomous multi-hop QA benchmark over HotpotQA distractor dev data.

This example is intentionally separate from ``open_dataset_qa.py``. It uses
HotpotQA, where each question ships with multiple Wikipedia paragraphs and
supporting-fact titles. The planner still compiles each graph autonomously, but
the evaluation is stricter than the SQuAD smoke test: retrieval must recover the
supporting titles and the final answer must contain the gold answer.

Default run:
    python examples/hotpot_multihop_qa.py --limit 3
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

from examples.open_dataset_qa import PromptDumpingProvider
from llmasm.api import LLMASM
from llmasm.analysis.visualize import to_viewer_graph
from llmasm.config import RuntimeConfig
from llmasm.errors import LLMASMError
from llmasm.graph.models import MemoryItem, WorkspaceEdge, WorkspaceEdgeType
from llmasm.graph.registry import default_schema_registry
from llmasm.ids import new_id
from llmasm.providers.ollama import OllamaProvider
from llmasm.schemas import RawText
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.base import ToolSpec
from llmasm.tools.registry import ToolRegistry

HOTPOT_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


@dataclass(frozen=True)
class HotpotArticle:
    """One article paragraph from a HotpotQA case."""

    title: str
    sentences: list[str]

    @property
    def text(self) -> str:
        return normalize_space(" ".join(self.sentences))


@dataclass(frozen=True)
class HotpotCase:
    """One selected multi-hop QA benchmark case."""

    id: str
    question: str
    answer: str
    question_type: str
    level: str
    articles: list[HotpotArticle]
    supporting_titles: list[str]


class HotpotContextSearchTool:
    """Search the candidate Wikipedia paragraphs for one HotpotQA case."""

    def __init__(self, case: HotpotCase, top_k: int = 5) -> None:
        self.case = case
        self.top_k = top_k

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="hotpot.search_context",
            description=(
                "Search the HotpotQA distractor context for evidence relevant to a multi-hop question. "
                "Input is RawText containing the user question. Output is RawText with the top candidate "
                "Wikipedia paragraphs. Use this before any model node for HotpotQA."
            ),
            input_schema="RawText",
            output_schema="RawText",
        )

    def invoke(self, input: BaseModel) -> BaseModel:
        query = extract_question(getattr(input, "text", ""))
        ranked = rank_articles(query, self.case.articles)[: self.top_k]
        blocks = [
            f"question: {query}",
            "Use at least two evidence paragraphs when the question requires a bridge or comparison.",
        ]
        for score, article in ranked:
            blocks.append(
                "\n".join(
                    [
                        f"title: {article.title}",
                        f"score: {score:.3f}",
                        "paragraph:",
                        article.text,
                    ]
                )
            )
        return RawText(text="\n\n---\n\n".join(blocks))


class WorkspaceNotesSearchTool:
    """Retrieve prior workspace memory for follow-up turns."""

    def __init__(self, storage: InMemoryStorage, workspace_id: str, limit: int = 4) -> None:
        self.storage = storage
        self.workspace_id = workspace_id
        self.limit = limit

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="workspace.search_notes",
            description=(
                "Search notes saved from previous calls in the current workspace. "
                "Input is RawText containing the follow-up request. Output is RawText "
                "with prior answers and evidence. Use this for follow-up questions that "
                "refer to the previous call, earlier answer, or prior evidence."
            ),
            input_schema="RawText",
            output_schema="RawText",
        )

    def invoke(self, input: BaseModel) -> BaseModel:
        query = getattr(input, "text", "")
        items = self.storage.search_memory(self.workspace_id, query, limit=self.limit)
        blocks = []
        for item in items:
            blocks.append(
                "\n".join(
                    [
                        f"memory_id: {item.id}",
                        f"kind: {item.kind}",
                        "text:",
                        item.text,
                    ]
                )
            )
        return RawText(text="\n\n---\n\n".join(blocks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="data/hotpot/hotpot_dev_distractor_v1.json")
    parser.add_argument("--download-url", default=HOTPOT_DEV_URL)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--conversation-demo",
        action="store_true",
        help="Run two dependent calls in one workspace and export a combined graph.",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=600)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--planner-model", default="gemma4:e4b")
    parser.add_argument("--runtime-model", default="gemma4:e4b")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--compiler-attempts", type=int, default=3)
    parser.add_argument("--report-jsonl", default="data/hotpot/llmasm_hotpot_report.jsonl")
    parser.add_argument("--summarize-report", default=None)
    parser.add_argument("--write-viewer-json", default=None)
    parser.add_argument("--debug-prompts", action="store_true")
    parser.add_argument("--debug-prompts-dir", default="data/hotpot/debug_prompts")
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

    cases = select_cases(load_cases(dataset_path, args.max_cases), args.limit, args.top_k)
    if len(cases) < args.limit:
        raise SystemExit(f"Only found {len(cases)} retrievable HotpotQA cases")
    if args.conversation_demo:
        run_conversation_demo(cases[0], args)
        return

    if args.retrieval_only:
        for index, case in enumerate(cases, start=1):
            output = HotpotContextSearchTool(case, args.top_k).invoke(RawText(text=case.question))
            support_recall = support_title_recall(output.text, case.supporting_titles)
            print(f"[{index}] support_title_recall={support_recall:.2f} type={case.question_type}")
            print(f"question: {case.question}")
            print(f"gold: {case.answer}")
            print(f"supporting_titles: {', '.join(case.supporting_titles)}")
        return

    report_path = Path(args.report_jsonl)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    last_analysis = None
    with report_path.open("w", encoding="utf-8") as report:
        for index, case in enumerate(cases, start=1):
            result = run_case(case, index, args)
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


def run_conversation_demo(case: HotpotCase, args: argparse.Namespace) -> None:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    storage = InMemoryStorage()
    tools.register(HotpotContextSearchTool(case, top_k=args.top_k))
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
    workspace_id = app.create_workspace("hotpot-followup-demo")
    first_prompt = (
        "Answer this HotpotQA multi-hop question using the registered HotpotQA context search tool. "
        "Retrieve evidence first, combine evidence from the relevant paragraphs, answer concisely, "
        "and do not use outside knowledge.\n\n"
        f"Question: {case.question}"
    )
    first_analysis = compile_and_run(app, workspace_id, first_prompt)
    first_answer = final_answer_text(first_analysis)
    first_retrieved = retrieved_text(first_analysis)
    memory = persist_turn_memory(storage, workspace_id, first_analysis.run.id, case, first_answer, first_retrieved)
    tools.register(WorkspaceNotesSearchTool(storage, workspace_id))
    second_prompt = (
        "This is a follow-up to the previous call. Use the registered workspace.search_notes tool "
        "to retrieve the previous answer and evidence, then identify the supporting article titles "
        "that were needed and restate the answer. Do not call the HotpotQA context search tool unless "
        "the prior notes are insufficient.\n\n"
        f"Follow-up question: Which previous evidence titles support the answer to '{case.question}'?"
    )
    second_analysis = compile_and_run(app, workspace_id, second_prompt)
    persist_context_edges(storage, workspace_id, first_analysis, second_analysis, memory)
    print("\nconversation_demo")
    print(f"question: {case.question}")
    print(f"gold: {case.answer}")
    print(f"first_answer: {first_answer[:500]}")
    print(f"second_answer: {final_answer_text(second_analysis)[:500]}")
    print(f"memory_id: {memory.id}")
    print(f"first_task_graph_id: {first_analysis.task_graph.id}")
    print(f"second_task_graph_id: {second_analysis.task_graph.id}")
    if args.write_viewer_json:
        target = Path(args.write_viewer_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(to_conversation_viewer_graph([first_analysis, second_analysis], [memory]), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"viewer_json: {target}")


def compile_and_run(app: LLMASM, workspace_id: str, prompt: str) -> Any:
    task_graph_id = app.compile(workspace_id, prompt)
    run_id = app.run(task_graph_id)
    return app.query_run(run_id)


def run_case(case: HotpotCase, index: int, args: argparse.Namespace) -> dict[str, Any]:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    tools.register(HotpotContextSearchTool(case, top_k=args.top_k))
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
    prompt = (
        "Answer this HotpotQA multi-hop question using the registered HotpotQA context search tool. "
        "Retrieve evidence first, combine evidence from the relevant paragraphs, answer concisely, "
        "and do not use outside knowledge.\n\n"
        f"Question: {case.question}"
    )
    workspace_id = app.create_workspace(f"hotpot-eval-{index}")
    started = time.perf_counter()
    try:
        task_graph_id = app.compile(workspace_id, prompt)
        run_id = app.run(task_graph_id)
        analysis = app.query_run(run_id)
        retrieved = retrieved_text(analysis)
        answer = final_answer_text(analysis)
        support_recall = support_title_recall(retrieved, case.supporting_titles)
        return {
            "case": index,
            "ok": True,
            "id": case.id,
            "question": case.question,
            "question_type": case.question_type,
            "level": case.level,
            "gold_answer": case.answer,
            "supporting_titles": case.supporting_titles,
            "task_graph_id": task_graph_id,
            "run_id": run_id,
            "compiled_nodes": len(analysis.task_graph.nodes),
            "tool_calls": len(analysis.tool_calls),
            "model_calls": len(analysis.model_calls),
            "support_title_recall": support_recall,
            "support_titles_complete": support_recall == 1.0,
            "answer_contains_gold": contains_answer(answer, case.answer),
            "answer": answer,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "analysis": analysis,
        }
    except LLMASMError as exc:
        return {
            "case": index,
            "ok": False,
            "id": case.id,
            "question": case.question,
            "question_type": case.question_type,
            "level": case.level,
            "gold_answer": case.answer,
            "supporting_titles": case.supporting_titles,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "last_errors": str(getattr(exc, "last_errors", "")),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }


def persist_turn_memory(
    storage: InMemoryStorage,
    workspace_id: str,
    run_id: str,
    case: HotpotCase,
    answer: str,
    retrieved: str,
) -> MemoryItem:
    memory = MemoryItem(
        id=new_id("memory"),
        workspace_graph_id=workspace_id,
        kind="qa_turn_summary",
        text="\n".join(
            [
                f"Question: {case.question}",
                f"Answer: {answer}",
                f"Supporting titles: {', '.join(case.supporting_titles)}",
                "Retrieved evidence:",
                retrieved[:4000],
            ]
        ),
        source_run_id=run_id,
        metadata={
            "case_id": case.id,
            "gold_answer": case.answer,
            "supporting_titles": case.supporting_titles,
        },
    )
    storage.persist_memory_item(memory)
    return memory


def persist_context_edges(
    storage: InMemoryStorage,
    workspace_id: str,
    first_analysis: Any,
    second_analysis: Any,
    memory: MemoryItem,
) -> None:
    first_final_id = final_node_id(first_analysis)
    second_root_id = second_analysis.task_graph.root_prompt_node_id or second_analysis.task_graph.nodes[0].id
    storage.persist_workspace_edge(
        WorkspaceEdge(
            id=new_id("edge"),
            workspace_graph_id=workspace_id,
            edge_type=WorkspaceEdgeType.PRODUCED,
            from_type="node",
            from_id=first_final_id,
            to_type="memory_item",
            to_id=memory.id,
            reason="first call saved as workspace memory",
        )
    )
    storage.persist_workspace_edge(
        WorkspaceEdge(
            id=new_id("edge"),
            workspace_graph_id=workspace_id,
            edge_type=WorkspaceEdgeType.USED_CONTEXT,
            from_type="memory_item",
            from_id=memory.id,
            to_type="node",
            to_id=second_root_id,
            reason="second call compiled with prior workspace context",
        )
    )


def to_conversation_viewer_graph(analyses: list[Any], memories: list[MemoryItem]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    workspace_edges: list[dict[str, Any]] = []
    for index, analysis in enumerate(analyses, start=1):
        graph = to_viewer_graph(analysis)
        for node in graph["nodes"]:
            node["metadata"] = {**node.get("metadata", {}), "conversation_turn": index}
            nodes.append(node)
        for edge in graph["edges"]:
            edge["metadata"] = {**edge.get("metadata", {}), "conversation_turn": index}
            edges.append(edge)
        workspace_edges.extend(graph.get("workspace_edges", []))
    for memory in memories:
        nodes.append(
            {
                "id": memory.id,
                "label": "workspace memory",
                "kind": "memory",
                "status": "persisted",
                "subtitle": memory.kind,
                "schema": {"input": None, "output": "RawText"},
                "metrics": {"artifacts": 1},
                "metadata": memory.model_dump(mode="json"),
            }
        )
    if analyses and memories:
        first_final = final_node_id(analyses[0])
        second_root = analyses[1].task_graph.root_prompt_node_id or analyses[1].task_graph.nodes[0].id
        memory = memories[0]
        edges.append(
            {
                "id": f"edge_{first_final}_{memory.id}",
                "source": first_final,
                "target": memory.id,
                "label": "produced memory",
                "type": "workspace",
                "required": True,
                "transform": None,
                "metadata": {"edge_type": "produced"},
            }
        )
        edges.append(
            {
                "id": f"edge_{memory.id}_{second_root}",
                "source": memory.id,
                "target": second_root,
                "label": "used context",
                "type": "workspace",
                "required": True,
                "transform": None,
                "metadata": {"edge_type": "used_context"},
            }
        )
    return {
        "metadata": {
            "workspace_id": analyses[0].workspace.id if analyses else None,
            "workspace_name": analyses[0].workspace.name if analyses else "",
            "task_graph_id": "conversation-demo",
            "run_id": "multi-run",
            "run_status": "succeeded",
            "turns": len(analyses),
        },
        "nodes": nodes,
        "edges": edges,
        "workspace_edges": workspace_edges,
    }


def final_node_id(analysis: Any) -> str:
    for node in analysis.task_graph.nodes:
        if node.kind == "final":
            return node.id
    return analysis.task_graph.nodes[-1].id


def ensure_dataset(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HotpotQA distractor dev data from {url}")
    urlretrieve(url, path)


def load_cases(path: Path, max_cases: int) -> list[HotpotCase]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    cases: list[HotpotCase] = []
    for row in rows[:max_cases]:
        articles = [
            HotpotArticle(title=title, sentences=list(sentences))
            for title, sentences in row["context"]
        ]
        support_titles = sorted({title for title, _ in row["supporting_facts"]})
        cases.append(
            HotpotCase(
                id=row["_id"],
                question=normalize_space(row["question"]),
                answer=normalize_space(row["answer"]),
                question_type=row.get("type", ""),
                level=row.get("level", ""),
                articles=articles,
                supporting_titles=support_titles,
            )
        )
    return cases


def select_cases(cases: list[HotpotCase], limit: int, top_k: int) -> list[HotpotCase]:
    selected: list[HotpotCase] = []
    for case in cases:
        if len(case.supporting_titles) < 2:
            continue
        if case.answer.lower() in {"yes", "no"}:
            continue
        retrieved = HotpotContextSearchTool(case, top_k).invoke(RawText(text=case.question)).text
        if support_title_recall(retrieved, case.supporting_titles) == 1.0 and contains_answer(
            retrieved, case.answer
        ):
            selected.append(case)
            if len(selected) >= limit:
                return selected
    return selected


def rank_articles(query: str, articles: list[HotpotArticle]) -> list[tuple[float, HotpotArticle]]:
    query_terms = weighted_terms(query)
    ranked = []
    for article in articles:
        text_terms = weighted_terms(article.title + " " + article.text)
        overlap = set(query_terms) & set(text_terms)
        if not query_terms or not text_terms:
            score = 0.0
        else:
            score = sum(query_terms[term] for term in overlap) / sum(query_terms.values())
        ranked.append((score, article))
    return sorted(ranked, key=lambda item: item[0], reverse=True)


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
            chunks.append(str(content.get("text", "")) if isinstance(content, dict) else str(content))
    return "\n".join(chunks)


def support_title_recall(text: str, supporting_titles: list[str]) -> float:
    if not supporting_titles:
        return 0.0
    normalized = normalize_for_match(text)
    hits = sum(1 for title in supporting_titles if normalize_for_match(title) in normalized)
    return hits / len(supporting_titles)


def print_result(result: dict[str, Any]) -> None:
    status = "ok" if result["ok"] else "failed"
    print(f"\n[{result['case']}] {status}: {result['question']}")
    print(f"type/level: {result['question_type']} / {result['level']}")
    print(f"gold: {result['gold_answer']}")
    print(f"supporting_titles: {', '.join(result['supporting_titles'])}")
    if result["ok"]:
        print(f"support_title_recall: {result['support_title_recall']:.2f}")
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
    support_complete = [row for row in ok_rows if row.get("support_titles_complete")]
    answer_hits = [row for row in ok_rows if row.get("answer_contains_gold")]
    elapsed = [float(row.get("elapsed_seconds") or 0) for row in rows]
    failure_counts: dict[str, int] = {}
    for row in rows:
        if row.get("ok"):
            continue
        key = str(row.get("error_type") or "unknown")
        failure_counts[key] = failure_counts.get(key, 0) + 1
    print("\nsummary")
    print(f"cases: {len(rows)}")
    print(f"ok: {len(ok_rows)}/{len(rows)}")
    print(f"support_titles_complete: {len(support_complete)}/{len(ok_rows)}")
    print(f"answer_contains_gold: {len(answer_hits)}/{len(ok_rows)}")
    print(f"avg_elapsed_seconds: {sum(elapsed) / len(elapsed):.2f}")
    if failure_counts:
        print("failures: " + ", ".join(f"{key}={value}" for key, value in sorted(failure_counts.items())))


def extract_question(text: str) -> str:
    match = re.search(r"(?is)\bquestion\s*:\s*(.+)", text)
    if match:
        return normalize_space(match.group(1))
    return normalize_space(text)


def contains_answer(text: str, answer: str) -> bool:
    return normalize_for_match(answer) in normalize_for_match(text)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def weighted_terms(text: str) -> dict[str, float]:
    terms: dict[str, float] = {}
    for token in re.findall(r"[A-Za-z0-9]+", text.lower()):
        if token in STOPWORDS or len(token) < 3:
            continue
        terms[token] = terms.get(token, 0.0) + 1.0
    return terms


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
    "whom",
    "whose",
    "did",
    "does",
    "have",
    "has",
    "had",
    "using",
    "use",
    "question",
    "answer",
    "registered",
    "tool",
    "evidence",
    "hotpotqa",
}


if __name__ == "__main__":
    main()
