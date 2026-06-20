"""Deterministic chat fast path — bypasses planner for conversation mode."""

from __future__ import annotations

from typing import Any

from llmasm.chunking import SentenceTextChunker
from llmasm.config import RuntimeConfig
from llmasm.conversation.classifier import DialogueType, classify_dialogue, classify_dialogue_llm
from llmasm.conversation.memory import (
    get_recent_qa_pairs,
    get_recent_user_questions,
    get_source_passages,
    store_assistant_answer,
    store_source_passage,
    store_source_passages,
    store_source_summary,
    store_system_note,
    store_user_question,
)
from llmasm.conversation.retrieval import (
    compose_instruction,
    retrieve_context,
)
from llmasm.graph.models import Node, NodeKind, Port, Run, TaskEdge, TaskGraph
from llmasm.ids import new_id
from llmasm.providers.base import LLMProvider
from llmasm.runtime.executor import Executor
from llmasm.schemas import FinalAnswer, Summary
from llmasm.storage.base import Storage
from llmasm.storage.embeddings import EmbeddingStore, NullEmbeddingStore
from llmasm.tools.registry import ToolRegistry


def _build_chat_graph(
    workspace_graph_id: str,
    prompt: str,
    instruction: str,
    model: str,
) -> TaskGraph:
    """Build a deterministic intent -> model -> final graph."""
    task_graph_id = new_id("taskgraph")
    node_intent_id = new_id("node")
    node_model_id = new_id("node")
    node_final_id = new_id("node")

    intent_node = Node(
        id=node_intent_id,
        workspace_graph_id=workspace_graph_id,
        task_graph_id=task_graph_id,
        kind=NodeKind.INTENT,
        name="intent",
        output_schema="RawText",
        ports=[
            Port(node_id=node_intent_id, name="output", direction="output", schema_ref="RawText"),
        ],
        execution={},
        metadata={"output": {"text": prompt}, "prompt": prompt},
    )

    model_node = Node(
        id=node_model_id,
        workspace_graph_id=workspace_graph_id,
        task_graph_id=task_graph_id,
        kind=NodeKind.MODEL,
        name="answer",
        input_schema="RawText",
        output_schema="FinalAnswer",
        ports=[
            Port(node_id=node_model_id, name="input", direction="input", schema_ref="RawText"),
            Port(node_id=node_model_id, name="output", direction="output", schema_ref="FinalAnswer"),
        ],
        execution={"provider": "ollama", "model": model, "allow_cache": False},
        metadata={
            "instruction": instruction,
            "grounding_mode": "strict",
        },
    )

    final_node = Node(
        id=node_final_id,
        workspace_graph_id=workspace_graph_id,
        task_graph_id=task_graph_id,
        kind=NodeKind.FINAL,
        name="final",
        input_schema="FinalAnswer",
        output_schema="FinalAnswer",
        ports=[
            Port(node_id=node_final_id, name="input", direction="input", schema_ref="FinalAnswer"),
            Port(node_id=node_final_id, name="output", direction="output", schema_ref="FinalAnswer"),
        ],
        execution={},
        metadata={},
    )

    edge_intent_model = TaskEdge(
        id=new_id("edge"),
        workspace_graph_id=workspace_graph_id,
        task_graph_id=task_graph_id,
        from_node_id=node_intent_id,
        from_port="output",
        to_node_id=node_model_id,
        to_port="input",
        required=True,
    )

    edge_model_final = TaskEdge(
        id=new_id("edge"),
        workspace_graph_id=workspace_graph_id,
        task_graph_id=task_graph_id,
        from_node_id=node_model_id,
        from_port="output",
        to_node_id=node_final_id,
        to_port="input",
        required=True,
    )

    return TaskGraph(
        id=task_graph_id,
        workspace_graph_id=workspace_graph_id,
        root_prompt_node_id=node_intent_id,
        nodes=[intent_node, model_node, final_node],
        task_edges=[edge_intent_model, edge_model_final],
        metadata={
            "intent": prompt,
            "goal_action": "continue",
            "fast_path": True,
        },
    )


def _build_summary_graph(
    workspace_graph_id: str,
    prompt: str,
    model: str,
) -> tuple[TaskGraph, str]:
    """Build a deterministic intent -> summary -> final graph.

    The summary model node produces a :class:`~llmasm.schemas.Summary` artifact
    that can be persisted as a workspace memory item.
    """

    task_graph_id = new_id("taskgraph")
    node_intent_id = new_id("node")
    node_summary_id = new_id("node")
    node_final_id = new_id("node")

    intent_node = Node(
        id=node_intent_id,
        workspace_graph_id=workspace_graph_id,
        task_graph_id=task_graph_id,
        kind=NodeKind.INTENT,
        name="intent",
        output_schema="RawText",
        ports=[
            Port(node_id=node_intent_id, name="output", direction="output", schema_ref="RawText"),
        ],
        execution={},
        metadata={"output": {"text": prompt}, "prompt": prompt},
    )

    summary_node = Node(
        id=node_summary_id,
        workspace_graph_id=workspace_graph_id,
        task_graph_id=task_graph_id,
        kind=NodeKind.MODEL,
        name="summary",
        input_schema="RawText",
        output_schema="Summary",
        ports=[
            Port(node_id=node_summary_id, name="input", direction="input", schema_ref="RawText"),
            Port(node_id=node_summary_id, name="output", direction="output", schema_ref="Summary"),
        ],
        execution={"provider": "ollama", "model": model, "allow_cache": False},
        metadata={
            "instruction": (
                "Summarize the source text below in a concise paragraph. "
                "Capture the main topics, named entities, and key claims."
            ),
            "grounding_mode": "permissive",
        },
    )

    final_node = Node(
        id=node_final_id,
        workspace_graph_id=workspace_graph_id,
        task_graph_id=task_graph_id,
        kind=NodeKind.FINAL,
        name="final",
        input_schema="Summary",
        output_schema="FinalAnswer",
        ports=[
            Port(node_id=node_final_id, name="input", direction="input", schema_ref="Summary"),
            Port(node_id=node_final_id, name="output", direction="output", schema_ref="FinalAnswer"),
        ],
        execution={},
        metadata={},
    )

    edge_intent_summary = TaskEdge(
        id=new_id("edge"),
        workspace_graph_id=workspace_graph_id,
        task_graph_id=task_graph_id,
        from_node_id=node_intent_id,
        from_port="output",
        to_node_id=node_summary_id,
        to_port="input",
        required=True,
    )

    edge_summary_final = TaskEdge(
        id=new_id("edge"),
        workspace_graph_id=workspace_graph_id,
        task_graph_id=task_graph_id,
        from_node_id=node_summary_id,
        from_port="output",
        to_node_id=node_final_id,
        to_port="input",
        required=True,
    )

    graph = TaskGraph(
        id=task_graph_id,
        workspace_graph_id=workspace_graph_id,
        root_prompt_node_id=node_intent_id,
        nodes=[intent_node, summary_node, final_node],
        task_edges=[edge_intent_summary, edge_summary_final],
        metadata={
            "intent": prompt,
            "goal_action": "continue",
            "fast_path": True,
            "summary_graph": True,
        },
    )
    return graph, node_summary_id


def _compose_instruction(
    prompt: str,
    dialogue_type: DialogueType,
    source_passages: list[str],
    recent_qa_pairs: list[tuple[str, str]],
    summary: str | None = None,
) -> str:
    """Compose a grounded-qa instruction for the model node."""
    parts: list[str] = []

    if source_passages:
        parts.append(
            "Answer using ONLY the source passages below. "
            "Base your answer on the information in the passages, including what they imply or contradict. "
            "If the passages contain no relevant information at all, say that the provided passage does not contain the answer. "
            "Do not use outside knowledge."
        )

        if summary:
            parts.append("\n--- Source summary ---")
            parts.append(summary)
            parts.append("---")

        parts.append("\n--- Source passages ---")
        for i, passage in enumerate(source_passages, 1):
            parts.append(f"Passage {i}:\n{passage}")
        parts.append("---\n")
    else:
        parts.append(
            "You are a helpful assistant. Answer the user's question based on the conversation context. "
            "If no relevant context is available, answer from your own knowledge."
        )

    if dialogue_type == DialogueType.FOLLOWUP_QUESTION and recent_qa_pairs:
        parts.append("\n--- Recent conversation (for resolving pronouns and follow-ups) ---")
        for q, a in recent_qa_pairs[:3]:
            parts.append(f"Q: {q}\nA: {a}")
        parts.append("---\n")

    parts.append(f"\nCurrent question: {prompt}")
    return "\n".join(parts)


def chat_turn(
    workspace_graph_id: str,
    prompt: str,
    *,
    storage: Storage,
    provider: LLMProvider,
    runtime_config: RuntimeConfig,
    tool_registry: ToolRegistry,
    schema_registry: Any,
    transform_registry: Any,
    embedding_store: EmbeddingStore | None = None,
    turn: int | None = None,
    out_info: dict[str, Any] | None = None,
) -> FinalAnswer:
    """Execute a single chat turn using the deterministic fast path.

    This bypasses the planner and compiler entirely for ordinary conversation
    and grounded QA. It creates a deterministic ``intent -> model -> final`` graph,
    executes it, and stores structured conversation memory.

    Args:
        out_info: Optional dict that will be populated with metadata about the
            turn, including ``instruction_tokens`` and ``run_id``.
    """
    # Use NullEmbeddingStore for chat memory unless chat embeddings are explicitly enabled.
    chat_embedding_store: EmbeddingStore = (
        embedding_store
        if runtime_config.chat_embeddings_enabled and embedding_store is not None
        else NullEmbeddingStore()
    )

    recent_questions = get_recent_user_questions(storage, workspace_graph_id, limit=5)
    classification_tokens = 0
    if runtime_config.llm_dialogue_classifier:
        dialogue_type, classification_tokens = classify_dialogue_llm(
            prompt,
            recent_questions,
            provider=provider,
            model=runtime_config.default_model,
        )
    else:
        dialogue_type = classify_dialogue(prompt, recent_questions)

    # Store source passages immediately when detected
    if dialogue_type == DialogueType.SOURCE:
        tokens = runtime_config.tokenizer.count_tokens(prompt)
        should_chunk = (
            runtime_config.chunking_enabled
            and tokens > runtime_config.chunking_trigger_tokens
        )
        if should_chunk:
            source_id = new_id("memory")
            chunker = SentenceTextChunker(
                runtime_config.tokenizer,
                target_tokens=runtime_config.chunk_target_tokens,
                overlap_tokens=runtime_config.chunk_overlap_tokens,
            )
            chunks = chunker.chunk(prompt, source_id=source_id)
            store_source_passages(
                workspace_graph_id,
                chunks,
                storage=storage,
                runtime_config=runtime_config,
                provider=provider,
                embedding_store=chat_embedding_store,
                turn=turn,
            )

            if runtime_config.chunking_summary_enabled:
                graph, summary_node_id = _build_summary_graph(
                    workspace_graph_id=workspace_graph_id,
                    prompt=prompt,
                    model=runtime_config.default_model,
                )
                storage.persist_task_graph(graph)

                run = Run(
                    id=new_id("run"),
                    workspace_graph_id=workspace_graph_id,
                    task_graph_id=graph.id,
                    metadata={
                        "user_prompt": prompt,
                        "fast_path": True,
                        "summary_for_source": source_id,
                    },
                )
                storage.create_run(run)

                executor = Executor(
                    storage=storage,
                    tool_registry=tool_registry,
                    provider=provider,
                    schema_registry=schema_registry,
                    transform_registry=transform_registry,
                    runtime_config=runtime_config,
                    embedding_store=embedding_store,
                )
                executor.execute(run.id)

                summary_artifacts = [
                    artifact
                    for artifact in storage.list_artifacts(run.id)
                    if artifact.node_id == summary_node_id and artifact.port == "output"
                ]
                if summary_artifacts:
                    summary_text = Summary.model_validate(
                        summary_artifacts[-1].content_json
                    ).text
                    store_source_summary(
                        workspace_graph_id,
                        summary_text,
                        source_id=source_id,
                        storage=storage,
                        runtime_config=runtime_config,
                        provider=provider,
                        embedding_store=chat_embedding_store,
                        source_run_id=run.id,
                        turn=turn,
                    )

            return FinalAnswer(text="Got it. I've saved the passage for questions.")

        store_source_passage(
            workspace_graph_id,
            prompt,
            storage=storage,
            runtime_config=runtime_config,
            provider=provider,
            embedding_store=chat_embedding_store,
            turn=turn,
        )
        return FinalAnswer(text="Got it. I've saved the passage for questions.")

    # For instruction-type prompts without a question, just acknowledge
    if dialogue_type == DialogueType.INSTRUCTION and not prompt.strip().endswith("?"):
        store_system_note(
            workspace_graph_id,
            prompt,
            storage=storage,
            runtime_config=runtime_config,
            provider=provider,
            embedding_store=chat_embedding_store,
            turn=turn,
        )
        return FinalAnswer(text="Understood. I'll keep that in mind for our conversation.")

    # Store the user question before answering
    store_user_question(
        workspace_graph_id,
        prompt,
        storage=storage,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=chat_embedding_store,
        turn=turn,
    )

    # Gather context for the model
    summary: str | None = None
    use_rag = runtime_config.chat_embeddings_enabled and not isinstance(
        chat_embedding_store, NullEmbeddingStore
    )
    if use_rag:
        summary, retrieved_passages, retrieved_qa_pairs, search_query = retrieve_context(
            workspace_graph_id,
            prompt,
            storage=storage,
            provider=provider,
            runtime_config=runtime_config,
            embedding_store=chat_embedding_store,
        )
        # Safety net: if retrieval returned nothing, fall back to all passages
        if not retrieved_passages:
            source_items = get_source_passages(storage, workspace_graph_id)
            summary = next(
                (item.text for item in source_items if item.metadata.get("is_summary")),
                None,
            )
            retrieved_passages = [
                item.text for item in source_items if not item.metadata.get("is_summary")
            ]
        instruction = compose_instruction(
            prompt=prompt,
            dialogue_type=dialogue_type.value,
            source_passages=retrieved_passages,
            qa_pairs=retrieved_qa_pairs,
            summary=summary,
        )
    else:
        source_items = get_source_passages(storage, workspace_graph_id)
        summary = next(
            (item.text for item in source_items if item.metadata.get("is_summary")),
            None,
        )
        retrieved_passages = [
            item.text for item in source_items if not item.metadata.get("is_summary")
        ]
        retrieved_qa_pairs = get_recent_qa_pairs(storage, workspace_graph_id, limit=3)
        instruction = _compose_instruction(
            prompt=prompt,
            dialogue_type=dialogue_type,
            source_passages=retrieved_passages,
            recent_qa_pairs=retrieved_qa_pairs,
            summary=summary,
        )
        search_query = ""
    instruction_tokens = runtime_config.tokenizer.count_tokens(instruction)

    graph = _build_chat_graph(
        workspace_graph_id=workspace_graph_id,
        prompt=prompt,
        instruction=instruction,
        model=runtime_config.default_model,
    )
    storage.persist_task_graph(graph)

    run = Run(
        id=new_id("run"),
        workspace_graph_id=workspace_graph_id,
        task_graph_id=graph.id,
        metadata={"user_prompt": prompt, "fast_path": True, "instruction_tokens": instruction_tokens},
    )
    storage.create_run(run)

    executor = Executor(
        storage=storage,
        tool_registry=tool_registry,
        provider=provider,
        schema_registry=schema_registry,
        transform_registry=transform_registry,
        runtime_config=runtime_config,
        embedding_store=embedding_store,
    )
    executor.execute(run.id)

    # Extract final answer
    final_node_ids = {node.id for node in graph.nodes if node.kind == NodeKind.FINAL}
    artifacts = [
        artifact
        for artifact in storage.list_artifacts(run.id)
        if artifact.node_id in final_node_ids
    ]
    if not artifacts:
        answer = FinalAnswer(text="", sources=[])
    else:
        answer = FinalAnswer.model_validate(artifacts[-1].content_json)

    # Store assistant answer
    store_assistant_answer(
        workspace_graph_id,
        answer.text,
        storage=storage,
        runtime_config=runtime_config,
        provider=provider,
        embedding_store=chat_embedding_store,
        source_run_id=run.id,
        turn=turn,
    )

    if out_info is not None:
        out_info["instruction_tokens"] = instruction_tokens + classification_tokens
        out_info["classification_tokens"] = classification_tokens
        out_info["run_id"] = run.id
        out_info["source_passage_count"] = len(retrieved_passages)
        out_info["recent_qa_pairs"] = len(retrieved_qa_pairs)
        out_info["search_query"] = search_query
        out_info["rag_enabled"] = use_rag
        out_info["summary_present"] = summary is not None

    return answer
