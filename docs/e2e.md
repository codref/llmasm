# End-To-End Testing

## Complex Ollama Example

The complex example uses a deterministic planner by default and your local
Ollama server for runtime model calls.

```bash
python examples/complex_end_to_end.py
```

Defaults:

- Ollama URL: `http://localhost:11434`
- Planner model advertised: `gemma4:26b`
- Runtime model: `gemma4:e4b`

You can try the real planner path too:

```bash
python examples/complex_end_to_end.py --planner ollama --timeout 600
```

The real planner path depends on the local model producing a valid
`TaskGraphProposal` JSON object. It is slower because the planner receives the
full schema for structured output. The static planner path is the recommended
smoke test because it keeps the graph deterministic while still testing real
Ollama model execution.

## Autonomous Open-Dataset QA

For a less deterministic benchmark, use the SQuAD 2.0 example:

```bash
python examples/open_dataset_qa.py --limit 3
```

This downloads the SQuAD 2.0 dev set from the official Stanford dataset URL,
builds a local lexical retriever over a subset of Wikipedia passages, selects
questions whose gold answer is retrievable, and asks the planner model to compile
a fresh graph for each prompt. No fixed planner proposal is supplied.

Useful options:

```bash
python examples/open_dataset_qa.py --download-only
python examples/open_dataset_qa.py --retrieval-only --limit 5
python examples/open_dataset_qa.py --limit 5 --max-paragraphs 800
python examples/open_dataset_qa.py --limit 1 --debug-prompts --timeout 30
python examples/open_dataset_qa.py --summarize-report data/squad/llmasm_qa_report.jsonl
python examples/open_dataset_qa.py \
  --limit 3 \
  --write-viewer-json docker/graph-viewer/data/graph.json
```

Before running the autonomous compile path, confirm Ollama is reachable:

```bash
curl http://localhost:11434/api/tags
```

If Ollama listens on a different address:

```bash
python examples/open_dataset_qa.py --ollama-url http://127.0.0.1:11434 --limit 3
```

`--debug-prompts` writes planner/runtime prompt files, response text, response
JSON, provider metadata, and timeout/error files under `data/squad/debug_prompts`.
In debug mode the example uses the configured `--planner-model` and
`--runtime-model` names instead of blocking on `/api/tags`, so prompt dumps are
still produced when the model-list endpoint is the source of the timeout.

The JSONL report defaults to:

```text
data/squad/llmasm_qa_report.jsonl
```

Each row records whether compilation/execution succeeded, whether retrieval
included the gold answer, whether the final answer contained the gold answer,
and the number of compiled nodes, tool calls, and model calls.

Local verification with `gemma4:e4b` selected seven SQuAD cases and all seven
compiled, ran, retrieved gold evidence, and returned answers containing the gold
answer.

SQuAD is distributed under the CC BY-SA 4.0 license. The dataset files are
downloaded under `data/`, which is ignored by git.

## Autonomous Multi-Hop QA

For a more complex autonomous benchmark, use the HotpotQA distractor example:

```bash
python examples/hotpot_multihop_qa.py --limit 3
```

This downloads the HotpotQA distractor dev set, selects answerable cases where
the local retriever can recover all supporting-fact titles, and asks the planner
model to compile a fresh graph for each multi-hop question. The example keeps a
separate report and debug directory from the SQuAD benchmark.

Useful options:

```bash
python examples/hotpot_multihop_qa.py --download-only
python examples/hotpot_multihop_qa.py --retrieval-only --limit 5
python examples/hotpot_multihop_qa.py \
  --limit 10 \
  --planner-model gemma4:e4b \
  --runtime-model gemma4:e4b \
  --debug-prompts \
  --timeout 120
python examples/hotpot_multihop_qa.py \
  --summarize-report data/hotpot/llmasm_hotpot_report.jsonl
python examples/hotpot_multihop_qa.py \
  --conversation-demo \
  --limit 1 \
  --planner-model gemma4:e4b \
  --runtime-model gemma4:e4b \
  --debug-prompts \
  --timeout 120 \
  --write-viewer-json docker/graph-viewer/data/graph.json
```

The report tracks graph success, supporting-title recall, final answer gold
matching, node counts, tool calls, model calls, and elapsed time. Debug prompts
are written under `data/hotpot/debug_prompts`.

`--conversation-demo` runs two dependent calls in one workspace. The first call
answers a HotpotQA question and persists the answer/evidence as workspace
memory. The second call is a follow-up that uses `workspace.search_notes` to
retrieve that prior memory. When `--write-viewer-json` is set, the viewer shows
both task graphs plus a `workspace memory` node connected by `produced memory`
and `used context` edges.

HotpotQA is introduced on the project homepage as a multi-hop question answering
dataset with supporting facts. The example downloads the official distractor dev
JSON from the dataset link used by the HotpotQA project.

Current local verification:

- `python examples/complex_end_to_end.py --timeout 240` succeeded against Ollama.
- It executed 7 nodes, 7 task edges, 2 tool calls, 3 model calls, 10 artifacts,
  and 14 checkpoints.
- `python examples/complex_end_to_end.py --planner ollama --timeout 240` used
  `gemma4:26b` for planning but timed out during the structured planner call.

## Postgres, pgvector, Apache AGE, And Graph Viewer

Build and start the local graph stack:

```bash
make e2e-stack-up
```

Connection defaults:

```text
postgresql://llmasm:llmasm@localhost:15432/llmasm
```

The host port defaults to `15432` to avoid colliding with a locally installed
Postgres on `5432`. Override it with `POSTGRES_PORT=...` if needed.

First-party graph viewer:

```text
http://localhost:3000
```

The graph viewer is a tiny dependency-free service built from
`docker/graph-viewer`. It serves `docker/graph-viewer/data/graph.json` and
renders it as an interactive SVG graph with pan, zoom, node dragging, filtering,
and an inspector.

To update the viewer with a fresh complex e2e run:

```bash
python examples/complex_end_to_end.py \
  --timeout 240 \
  --write-viewer-json docker/graph-viewer/data/graph.json
```

Then click `Reload` in the viewer.

The Python example can also export static formats:

```bash
python examples/complex_end_to_end.py \
  --timeout 240 \
  --write-mermaid /tmp/llmasm.mmd \
  --write-dot /tmp/llmasm.dot
```

The Postgres init scripts still seed `llmasm_graph` with the deterministic
complex e2e task graph for AGE experiments. You can inspect it from psql:

```bash
docker exec llmasm-postgres psql -U llmasm -d llmasm -c \
  "LOAD 'age'; SET search_path = ag_catalog, public; \
   SELECT * FROM cypher('llmasm_graph', $$ MATCH (a)-[e]->(b) RETURN count(e) $$) AS (edges agtype);"
```

To stop the stack:

```bash
make e2e-stack-down
```

To force Postgres to rerun the init scripts and reseed the AGE graph:

```bash
docker compose down -v
make e2e-stack-up
```

The image starts from `pgvector/pgvector:pg16`, builds Apache AGE from the
`AGE_REF` build argument, creates the `vector` and `age` extensions, and creates
an AGE graph named `llmasm_graph`.

Apache AGE remains useful as a storage/query backend, but the old AGE Viewer UI
is not part of the e2e stack because its current dependency tree is brittle.

If the Apache AGE branch name changes for your target Postgres version, pass a
different ref:

```bash
AGE_REF=PG16 docker compose build postgres
```
