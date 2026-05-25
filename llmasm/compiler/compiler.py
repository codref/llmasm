"""Prompt-to-task-graph compiler."""

from __future__ import annotations

import json
from typing import Any

from llmasm.compiler.prompt import PriorContext, render_planner_prompt
from llmasm.compiler.proposal import ProposalEdge, ProposalNode, TaskGraphProposal
from llmasm.compiler.repair import compile_with_repair
from llmasm.config import RuntimeConfig
from llmasm.errors import CompilationError
from llmasm.goals.classifier import classify_goal_action
from llmasm.goals.tracker import GoalTracker
from llmasm.graph.models import (
    GoalAction,
    Node,
    NodeKind,
    Port,
    TaskEdge,
    TaskGraph,
    WorkspaceEdge,
)
from llmasm.graph.registry import SchemaRegistry
from llmasm.graph.transforms import TransformRegistry
from llmasm.graph.validation import (
    ValidationIssue,
    validate_acyclic,
    validate_context_budgets,
    validate_edge_compatibility,
    validate_models,
    validate_proposal_goal_action,
    validate_required_ports,
    validate_schema_refs,
    validate_terminal_node,
    validate_tools,
)
from llmasm.ids import new_id
from llmasm.providers.base import LLMProvider
from llmasm.storage.base import Storage
from llmasm.tools.registry import ToolRegistry


class Compiler:
    """Compile prompts into persisted task graphs."""

    def __init__(
        self,
        *,
        storage: Storage,
        planner: LLMProvider,
        schema_registry: SchemaRegistry,
        transform_registry: TransformRegistry,
        tool_registry: ToolRegistry,
        runtime_config: RuntimeConfig,
    ) -> None:
        self.storage = storage
        self.planner = planner
        self.schema_registry = schema_registry
        self.transform_registry = transform_registry
        self.tool_registry = tool_registry
        self.runtime_config = runtime_config
        self.goal_tracker = GoalTracker(storage)

    def compile_into_workspace(self, workspace_graph_id: str, prompt: str) -> str:
        """Compile a prompt into a persisted task graph and return its ID."""

        self.storage.load_workspace_graph(workspace_graph_id)
        active_goal = self.goal_tracker.load_active_goal(workspace_graph_id)
        goal_action = classify_goal_action(prompt, active_goal)
        goal = (
            self.goal_tracker.create_provisional_goal(workspace_graph_id, prompt)
            if goal_action == GoalAction.NEW
            else active_goal
        )
        few_shots = self.storage.retrieve_few_shot_examples(workspace_graph_id, prompt, 2)
        context = self._prior_context(workspace_graph_id, prompt)
        planner_prompt = render_planner_prompt(
            schema_registry=self.schema_registry,
            tool_registry=self.tool_registry,
            models=self.planner.list_models(),
            active_goal=goal,
            goal_action=goal_action,
            prior_context=context,
            user_prompt=prompt,
            runtime_config=self.runtime_config,
            few_shot_examples=few_shots,
        )
        try:
            proposal = compile_with_repair(
                self.planner,
                planner_prompt,
                goal_action.value,
                self.runtime_config.compiler_max_attempts,
                self._validate_proposal,
                {"model": self.runtime_config.planner_model},
            )
        except CompilationError as exc:
            self.storage.persist_compilation_failure(
                workspace_graph_id,
                {
                    "prompt": prompt,
                    "errors": str(exc.last_errors),
                    "raw": exc.last_raw_output,
                },
            )
            raise

        task_graph = self._normalize(workspace_graph_id, proposal, prompt, goal.id if goal else None)
        self.storage.persist_task_graph(task_graph)
        for edge in self._workspace_edges(task_graph, proposal):
            self.storage.persist_workspace_edge(edge)
        if goal is not None:
            if goal_action == GoalAction.NEW:
                self.goal_tracker.finalize_goal(
                    goal.id,
                    proposal.goal_update_text or prompt,
                    active_task_graph_id=task_graph.id,
                )
            elif goal_action == GoalAction.STEER:
                self.goal_tracker.steer_goal(goal.id, proposal.goal_update_text or goal.text)
        return task_graph.id

    def _prior_context(self, workspace_graph_id: str, prompt: str) -> list[PriorContext]:
        storage = self.storage
        if hasattr(storage, "retrieve_workspace_context"):
            items = storage.retrieve_workspace_context(workspace_graph_id, prompt, 800, {})
            return [PriorContext(item.kind, item.text) for item in items]
        return []

    def _validate_proposal(
        self, proposal: TaskGraphProposal, expected_goal_action: str
    ) -> list[ValidationIssue]:
        issues = validate_proposal_goal_action(proposal, expected_goal_action)
        graph = self._normalize("workspace_validation", proposal, "", None, dry_run=True)
        issues.extend(validate_schema_refs(graph, self.schema_registry))
        issues.extend(validate_edge_compatibility(graph, self.transform_registry))
        issues.extend(validate_tools(graph, self.tool_registry))
        issues.extend(validate_models(graph, self.planner.list_models()))
        issues.extend(validate_context_budgets(graph))
        issues.extend(validate_required_ports(graph))
        issues.extend(validate_terminal_node(graph))
        issues.extend(validate_acyclic(graph))
        return issues

    def _normalize(
        self,
        workspace_graph_id: str,
        proposal: TaskGraphProposal,
        prompt: str,
        goal_id: str | None,
        *,
        dry_run: bool = False,
    ) -> TaskGraph:
        task_graph_id = "taskgraph_validation" if dry_run else new_id("taskgraph")
        name_to_id: dict[str, str] = {}
        nodes: list[Node] = []
        if not any(node.kind == NodeKind.INTENT for node in proposal.nodes):
            prompt_node = ProposalNode(
                name="root_prompt",
                kind=NodeKind.INTENT,
                output_schema="RawText",
                metadata={"prompt": prompt},
            )
            proposal_nodes = [prompt_node, *proposal.nodes]
        else:
            proposal_nodes = proposal.nodes
        for proposed in proposal_nodes:
            node_id = f"node_{proposed.name}" if dry_run else new_id("node")
            name_to_id[proposed.name] = node_id
            input_schema, output_schema, execution, metadata = self._canonical_node_fields(proposed)
            ports = [
                Port(
                    node_id=node_id,
                    name=port.name,
                    direction=port.direction,  # type: ignore[arg-type]
                    schema_ref=port.schema_ref,
                    required=port.required,
                )
                for port in proposed.ports
            ]
            if input_schema and not any(port.direction == "input" for port in ports):
                ports.append(
                    Port(
                        node_id=node_id,
                        name="input",
                        direction="input",
                        schema_ref=input_schema,
                    )
                )
            if output_schema and not any(port.direction == "output" for port in ports):
                ports.append(
                    Port(
                        node_id=node_id,
                        name="output",
                        direction="output",
                        schema_ref=output_schema,
                    )
                )
            node = Node(
                id=node_id,
                workspace_graph_id=workspace_graph_id,
                task_graph_id=task_graph_id,
                kind=proposed.kind,
                name=proposed.name,
                input_schema=input_schema,
                output_schema=output_schema,
                ports=ports,
                execution=execution,
                metadata={**metadata, "proposal_name": proposed.name},
            )
            nodes.append(node)
        proposal_edges = self._canonical_edges(proposal_nodes, proposal.edges)
        edges = [
            TaskEdge(
                id=f"edge_{edge.from_node}_{edge.to_node}" if dry_run else new_id("edge"),
                workspace_graph_id=workspace_graph_id,
                task_graph_id=task_graph_id,
                from_node_id=name_to_id[edge.from_node],
                from_port=edge.from_port,
                to_node_id=name_to_id[edge.to_node],
                to_port=edge.to_port,
                transform=edge.transform,
                required=edge.required,
                metadata=edge.metadata,
            )
            for edge in proposal_edges
            if edge.from_node in name_to_id and edge.to_node in name_to_id
        ]
        metadata: dict[str, Any] = {
            **proposal.metadata,
            "intent": proposal.intent,
            "goal_action": proposal.goal_action.value,
            "goal_id": goal_id,
            "proposal_json": json.dumps(proposal.model_dump(mode="json"), sort_keys=True),
        }
        return TaskGraph(
            id=task_graph_id,
            workspace_graph_id=workspace_graph_id,
            root_prompt_node_id=nodes[0].id if nodes else None,
            nodes=nodes,
            task_edges=edges,
            metadata=metadata,
        )

    def _canonical_node_fields(
        self, proposed: ProposalNode
    ) -> tuple[str | None, str | None, dict[str, Any], dict[str, Any]]:
        metadata = dict(proposed.metadata)
        execution = proposed.execution.model_dump(exclude_none=True)
        input_schema = proposed.input_schema or self._metadata_string(metadata, "input_schema")
        output_schema = proposed.output_schema or self._metadata_string(metadata, "output_schema")
        if proposed.kind == NodeKind.INTENT:
            output_schema = output_schema or "RawText"
        if proposed.kind == NodeKind.FINAL:
            input_schema = input_schema or "Summary"
            output_schema = output_schema or "FinalAnswer"
        if proposed.kind in {NodeKind.MODEL, NodeKind.COMPRESS}:
            input_schema = input_schema or "RawText"
        if proposed.kind == NodeKind.TOOL:
            tool_name = execution.get("tool")
            if not tool_name:
                tool_name = self._infer_tool_name(proposed.name)
                if tool_name:
                    execution["tool"] = tool_name
            if tool_name and self.tool_registry.has(str(tool_name)):
                spec = self.tool_registry.get(str(tool_name)).spec()
                input_schema = input_schema or spec.input_schema
                output_schema = output_schema or spec.output_schema
        return input_schema, output_schema, execution, metadata

    def _metadata_string(self, metadata: dict[str, Any], key: str) -> str | None:
        value = metadata.get(key)
        return value if isinstance(value, str) else None

    def _infer_tool_name(self, node_name: str) -> str | None:
        tool_names = self.tool_registry.names()
        if node_name in tool_names:
            return node_name
        normalized_node = self._normalized_tool_alias(node_name)
        matches = [
            name for name in tool_names if self._normalized_tool_alias(name) == normalized_node
        ]
        if len(matches) == 1:
            return matches[0]
        if len(tool_names) == 1:
            return tool_names[0]
        return None

    def _normalized_tool_alias(self, value: str) -> str:
        alias = "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")
        parts = alias.split("_")
        if parts and parts[-1].isdigit():
            alias = "_".join(parts[:-1])
        return alias

    def _canonical_edges(
        self, nodes: list[ProposalNode], edges: list[Any]
    ) -> list[Any]:
        canonical = list(edges)
        intent = self._single_node(nodes, NodeKind.INTENT)
        tool = self._single_node(nodes, NodeKind.TOOL)
        model = self._single_node(nodes, NodeKind.MODEL)
        final = self._single_node(nodes, NodeKind.FINAL)
        if not (intent and tool and model and final):
            return canonical
        tool_input, tool_output, _, _ = self._canonical_node_fields(tool)
        model_input, _, _, _ = self._canonical_node_fields(model)
        if tool_input != "RawText" or tool_output != "RawText" or model_input != "RawText":
            return canonical
        canonical = [
            edge
            for edge in canonical
            if not (
                edge.from_node == intent.name
                and edge.to_node in {model.name, final.name}
            )
        ]
        self._ensure_edge(canonical, intent.name, tool.name)
        self._ensure_edge(canonical, tool.name, model.name)
        self._ensure_edge(canonical, model.name, final.name)
        return canonical

    def _single_node(self, nodes: list[ProposalNode], kind: NodeKind) -> ProposalNode | None:
        matching = [node for node in nodes if node.kind == kind]
        return matching[0] if len(matching) == 1 else None

    def _ensure_edge(self, edges: list[Any], from_node: str, to_node: str) -> None:
        if any(edge.from_node == from_node and edge.to_node == to_node for edge in edges):
            return
        edges.append(ProposalEdge(from_node=from_node, to_node=to_node))

    def _workspace_edges(
        self, task_graph: TaskGraph, proposal: TaskGraphProposal
    ) -> list[WorkspaceEdge]:
        by_name = {node.name: node.id for node in task_graph.nodes}
        edges: list[WorkspaceEdge] = []
        goal_id = task_graph.metadata.get("goal_id")
        if goal_id:
            edges.append(
                WorkspaceEdge(
                    id=new_id("edge"),
                    workspace_graph_id=task_graph.workspace_graph_id,
                    edge_type="supports_goal",  # type: ignore[arg-type]
                    from_type="task_graph",
                    from_id=task_graph.id,
                    to_type="goal",
                    to_id=str(goal_id),
                    reason="Compiled task graph supports active goal",
                )
            )
        for link in proposal.workspace_links:
            from_id = link.from_id or (by_name.get(link.from_node or "") if link.from_node else None)
            to_id = link.to_id or (by_name.get(link.to_node or "") if link.to_node else None)
            if from_id and to_id:
                edges.append(
                    WorkspaceEdge(
                        id=new_id("edge"),
                        workspace_graph_id=task_graph.workspace_graph_id,
                        edge_type=link.edge_type,
                        from_type=link.from_type,
                        from_id=from_id,
                        to_type=link.to_type,
                        to_id=to_id,
                        reason=link.reason,
                        metadata=link.metadata,
                    )
                )
        return edges
