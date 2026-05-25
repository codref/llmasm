"""Complex end-to-end example using Ollama for model execution.

Default mode uses a deterministic planner proposal and Ollama for runtime model
nodes. This keeps the graph shape stable while still exercising real local LLM
calls, tools, persistence, checkpoints, and run analysis.

Try:
    python examples/complex_end_to_end.py
    python examples/complex_end_to_end.py --planner ollama
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmasm.api import LLMASM
from llmasm.analysis.visualize import to_dot, to_mermaid, to_viewer_graph
from llmasm.config import RuntimeConfig
from llmasm.errors import CompilationError
from llmasm.graph.registry import default_schema_registry
from llmasm.providers.base import EmbeddingOutput, ModelInfo, ModelOutput
from llmasm.providers.ollama import OllamaProvider
from llmasm.schemas import RawText
from llmasm.storage.memory import InMemoryStorage
from llmasm.tools.base import ToolSpec
from llmasm.tools.registry import ToolRegistry


class StaticPlannerWithOllamaRuntime:
    """Return one known proposal for planning and delegate runtime calls to Ollama."""

    name = "static+ollama"

    def __init__(self, ollama: OllamaProvider, proposal_json: str, model_names: list[str]) -> None:
        self.ollama = ollama
        self.proposal_json = proposal_json
        self.model_names = model_names

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name=name, context_window=8192) for name in self.model_names]

    def generate(
        self,
        prompt: str,
        options: dict[str, Any] | None = None,
        format_schema: dict[str, Any] | None = None,
    ) -> ModelOutput:
        if format_schema is not None:
            return ModelOutput(text=self.proposal_json)
        return self.ollama.generate(prompt, options, format_schema)

    def embed(
        self,
        texts: list[str],
        options: dict[str, Any] | None = None,
    ) -> list[EmbeddingOutput]:
        return self.ollama.embed(texts, options)


class IncidentDocSearchTool:
    """Search a tiny local incident/document corpus."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="corpus.search_incident_docs",
            description="Search product incident docs and quarterly planning notes.",
            input_schema="RawText",
            output_schema="RawText",
        )

    def invoke(self, input: BaseModel) -> BaseModel:
        query = getattr(input, "text", "")
        return RawText(
            text=(
                f"Query: {query}\n\n"
                "Incident INC-1842: EU checkout latency rose between 2026-05-18 and "
                "2026-05-20. Root cause was connection pool exhaustion in payment-gateway-v2. "
                "Mitigation: cap payment retries, add queue-depth alerts, and ship pool sizing "
                "config per region.\n\n"
                "Quarterly planning note: Q3 objective is reduce checkout p95 by 20%, preserve "
                "conversion, and avoid schema churn in order-service. Owners: Payments, Platform, "
                "Data Reliability.\n\n"
                "Architecture note: checkout path depends on payment-gateway-v2, fraud-score, "
                "inventory-reservation, and order-service."
            )
        )


class TicketSearchTool:
    """Search a tiny local ticket corpus."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="corpus.search_tickets",
            description="Search open engineering tickets and risk items.",
            input_schema="RawText",
            output_schema="RawText",
        )

    def invoke(self, input: BaseModel) -> BaseModel:
        query = getattr(input, "text", "")
        return RawText(
            text=(
                f"Query: {query}\n\n"
                "PAY-771 open: Retry cap rollout needs load-test approval. Risk: reducing retries "
                "can expose transient PSP errors unless the client shows clear recovery guidance.\n"
                "PLAT-312 open: Add queue-depth and connection saturation dashboards by region.\n"
                "DATA-119 open: Backfill incident labels for May checkout events so analytics can "
                "separate payment latency from fraud-score latency.\n"
                "ORD-208 deferred: order-service schema simplification moved to Q4."
            )
        )


def build_proposal(model: str) -> str:
    """Return a complex but deterministic planner proposal."""

    return json.dumps(
        {
            "intent": "produce incident-informed Q3 execution brief",
            "goal_action": "new",
            "goal_update_text": "Create an incident-informed Q3 checkout execution brief.",
            "nodes": [
                {
                    "name": "intent",
                    "kind": "intent",
                    "output_schema": "RawText",
                    "metadata": {
                        "output": {
                            "text": (
                                "Build an executive brief for Q3 checkout reliability using the "
                                "May EU checkout incident, open tickets, owners, risks, and next actions."
                            )
                        }
                    },
                },
                {
                    "name": "retrieve_incident_docs",
                    "kind": "tool",
                    "input_schema": "RawText",
                    "output_schema": "RawText",
                    "execution": {"tool": "corpus.search_incident_docs", "allow_cache": True},
                },
                {
                    "name": "retrieve_tickets",
                    "kind": "tool",
                    "input_schema": "RawText",
                    "output_schema": "RawText",
                    "execution": {"tool": "corpus.search_tickets", "allow_cache": True},
                },
                {
                    "name": "extract_incident_facts",
                    "kind": "model",
                    "output_schema": "Summary",
                    "ports": [
                        {
                            "name": "incident_docs",
                            "direction": "input",
                            "schema_ref": "RawText",
                        },
                        {"name": "output", "direction": "output", "schema_ref": "Summary"},
                    ],
                    "execution": {"provider": "ollama", "model": model, "allow_cache": False},
                    "metadata": {
                        "instruction": (
                            "Extract only concrete facts from the incident docs. Include dates, "
                            "systems, root cause, mitigation, and owners. Be concise."
                        ),
                        "max_input_tokens": 2500,
                    },
                },
                {
                    "name": "extract_ticket_risks",
                    "kind": "model",
                    "output_schema": "Summary",
                    "ports": [
                        {"name": "tickets", "direction": "input", "schema_ref": "RawText"},
                        {"name": "output", "direction": "output", "schema_ref": "Summary"},
                    ],
                    "execution": {"provider": "ollama", "model": model, "allow_cache": False},
                    "metadata": {
                        "instruction": (
                            "Extract risks, blockers, and owner actions from the ticket list. "
                            "Group by team and keep the result brief."
                        ),
                        "max_input_tokens": 2500,
                    },
                },
                {
                    "name": "synthesize_execution_brief",
                    "kind": "model",
                    "output_schema": "Summary",
                    "ports": [
                        {
                            "name": "incident_facts",
                            "direction": "input",
                            "schema_ref": "Summary",
                        },
                        {
                            "name": "ticket_risks",
                            "direction": "input",
                            "schema_ref": "Summary",
                        },
                        {"name": "output", "direction": "output", "schema_ref": "Summary"},
                    ],
                    "execution": {"provider": "ollama", "model": model, "allow_cache": False},
                    "metadata": {
                        "instruction": (
                            "Write a crisp executive brief with sections: Summary, Evidence, "
                            "Risks, Recommended next actions, and Open questions. Use only "
                            "the supplied inputs."
                        ),
                        "max_input_tokens": 3500,
                    },
                },
                {
                    "name": "final",
                    "kind": "final",
                    "input_schema": "Summary",
                    "output_schema": "FinalAnswer",
                },
            ],
            "edges": [
                {
                    "from_node": "intent",
                    "from_port": "output",
                    "to_node": "retrieve_incident_docs",
                    "to_port": "input",
                },
                {
                    "from_node": "intent",
                    "from_port": "output",
                    "to_node": "retrieve_tickets",
                    "to_port": "input",
                },
                {
                    "from_node": "retrieve_incident_docs",
                    "from_port": "output",
                    "to_node": "extract_incident_facts",
                    "to_port": "incident_docs",
                },
                {
                    "from_node": "retrieve_tickets",
                    "from_port": "output",
                    "to_node": "extract_ticket_risks",
                    "to_port": "tickets",
                },
                {
                    "from_node": "extract_incident_facts",
                    "from_port": "output",
                    "to_node": "synthesize_execution_brief",
                    "to_port": "incident_facts",
                },
                {
                    "from_node": "extract_ticket_risks",
                    "from_port": "output",
                    "to_node": "synthesize_execution_brief",
                    "to_port": "ticket_risks",
                },
                {
                    "from_node": "synthesize_execution_brief",
                    "from_port": "output",
                    "to_node": "final",
                    "to_port": "input",
                },
            ],
        }
    )


def build_app(args: argparse.Namespace) -> LLMASM:
    schemas = default_schema_registry()
    tools = ToolRegistry(schemas)
    tools.register(IncidentDocSearchTool())
    tools.register(TicketSearchTool())

    ollama = OllamaProvider(
        base_url=args.ollama_url,
        timeout=args.timeout,
        default_model=args.runtime_model,
        embedding_model=args.embedding_model,
    )
    if args.planner == "ollama":
        provider = ollama
    else:
        provider = StaticPlannerWithOllamaRuntime(
            ollama=ollama,
            proposal_json=build_proposal(args.runtime_model),
            model_names=[args.runtime_model, args.planner_model],
        )
    return LLMASM(
        storage=InMemoryStorage(),
        provider=provider,
        tool_registry=tools,
        schema_registry=schemas,
        runtime_config=RuntimeConfig(
            planner_model=args.planner_model,
            default_model=args.runtime_model,
            compiler_max_attempts=args.compiler_attempts,
            embeddings_enabled=False,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--planner", choices=["static", "ollama"], default="static")
    parser.add_argument("--planner-model", default="gemma4:26b")
    parser.add_argument("--runtime-model", default="gemma4:e4b")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--compiler-attempts", type=int, default=3)
    parser.add_argument(
        "--write-viewer-json",
        default=None,
        help="Write first-party graph viewer JSON to this path.",
    )
    parser.add_argument("--write-mermaid", default=None, help="Write Mermaid flowchart to this path.")
    parser.add_argument("--write-dot", default=None, help="Write Graphviz DOT to this path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)
    workspace_id = app.create_workspace("complex-e2e", target_provider="ollama")
    prompt = (
        "Using the May EU checkout incident and open engineering tickets, produce a Q3 "
        "execution brief with evidence, risks, owners, recommended next actions, and "
        "open questions."
    )
    try:
        task_graph_id = app.compile(workspace_id, prompt)
    except CompilationError as exc:
        print("Planner failed to compile a valid TaskGraphProposal.")
        print(f"attempts:        {exc.attempts}")
        print(f"last_errors:     {exc.last_errors}")
        print(f"last_raw_output: {exc.last_raw_output}")
        raise SystemExit(2) from exc
    run_id = app.run(task_graph_id)
    analysis = app.query_run(run_id)
    answer_artifact = [
        artifact
        for artifact in analysis.artifacts
        if artifact.node_id in {node.id for node in analysis.task_graph.nodes if node.kind == "final"}
    ][-1]

    print("\n=== Final Answer ===\n")
    print(answer_artifact.content_json["text"])
    print("\n=== Run Analysis ===")
    print(f"task_graph_id: {task_graph_id}")
    print(f"run_id:        {run_id}")
    print(f"nodes:         {len(analysis.task_graph.nodes)}")
    print(f"task_edges:    {len(analysis.task_edges)}")
    print(f"artifacts:     {len(analysis.artifacts)}")
    print(f"tool_calls:    {len(analysis.tool_calls)}")
    print(f"model_calls:   {len(analysis.model_calls)}")
    print(f"checkpoints:   {len(analysis.checkpoints)}")
    print(f"token_usage:   {analysis.token_usage()}")

    if args.write_viewer_json:
        write_text(args.write_viewer_json, json.dumps(to_viewer_graph(analysis), indent=2))
        print(f"viewer_json:   {args.write_viewer_json}")
    if args.write_mermaid:
        write_text(args.write_mermaid, to_mermaid(analysis))
        print(f"mermaid:       {args.write_mermaid}")
    if args.write_dot:
        write_text(args.write_dot, to_dot(analysis))
        print(f"dot:           {args.write_dot}")


def write_text(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
