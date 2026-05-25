create table if not exists workspace_graphs (
  id text primary key,
  name text not null,
  status text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null
);

create table if not exists task_graphs (
  id text primary key,
  workspace_graph_id text not null references workspace_graphs(id),
  root_prompt_node_id text,
  parent_task_graph_id text,
  status text not null,
  compiler_version text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null
);

create table if not exists runs (
  id text primary key,
  workspace_graph_id text not null references workspace_graphs(id),
  task_graph_id text not null references task_graphs(id),
  status text not null,
  program_counter_node_id text,
  metadata jsonb not null default '{}'::jsonb,
  started_at timestamptz,
  completed_at timestamptz
);

create table if not exists nodes (
  id text primary key,
  workspace_graph_id text not null,
  task_graph_id text not null references task_graphs(id),
  kind text not null,
  name text not null,
  input_schema text,
  output_schema text,
  ports_json jsonb not null default '[]'::jsonb,
  execution_json jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null
);

create table if not exists task_edges (
  id text primary key,
  workspace_graph_id text not null,
  task_graph_id text not null references task_graphs(id),
  from_node_id text not null,
  from_port text not null,
  to_node_id text not null,
  to_port text not null,
  transform text,
  required boolean not null,
  metadata jsonb not null default '{}'::jsonb
);

create table if not exists workspace_edges (
  id text primary key,
  workspace_graph_id text not null,
  edge_type text not null,
  from_type text not null,
  from_id text not null,
  from_port text,
  to_type text not null,
  to_id text not null,
  to_port text,
  reason text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null
);

create table if not exists run_node_states (
  run_id text not null references runs(id),
  node_id text not null,
  status text not null,
  attempts integer not null,
  last_error_json jsonb,
  output_artifact_ids jsonb not null default '[]'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null,
  primary key (run_id, node_id)
);

create table if not exists artifacts (
  id text primary key,
  run_id text not null references runs(id),
  node_id text not null,
  port text not null,
  content_type text not null,
  content_json jsonb,
  content_ref text,
  token_count integer not null,
  superseded_by text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null
);

create table if not exists tool_calls (
  id text primary key,
  run_id text not null references runs(id),
  node_id text not null,
  tool_name text not null,
  input_json jsonb,
  output_artifact_id text,
  status text not null,
  latency_ms integer,
  created_at timestamptz not null
);

create table if not exists model_calls (
  id text primary key,
  run_id text not null references runs(id),
  node_id text not null,
  provider text not null,
  model text not null,
  prompt_artifact_id text,
  output_artifact_id text,
  status text not null,
  token_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null
);

create table if not exists goals (
  id text primary key,
  workspace_graph_id text not null,
  active_task_graph_id text,
  text text not null,
  status text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null,
  updated_at timestamptz not null
);

create table if not exists memory_items (
  id text primary key,
  workspace_graph_id text not null,
  kind text not null,
  text text not null,
  source_artifact_id text,
  source_run_id text,
  confidence double precision not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null
);

create table if not exists embeddings (
  id text primary key,
  owner_type text not null,
  owner_id text not null,
  model text not null,
  dimensions integer not null,
  text_hash text not null,
  vector_json jsonb,
  created_at timestamptz not null
);

create table if not exists checkpoints (
  id text primary key,
  run_id text not null references runs(id),
  program_counter_node_id text,
  completed_node_ids jsonb not null default '[]'::jsonb,
  failed_node_ids jsonb not null default '[]'::jsonb,
  state_hash text not null,
  state_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null
);

create table if not exists compilation_failures (
  id bigserial primary key,
  workspace_graph_id text not null,
  payload jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists idx_task_graphs_workspace on task_graphs(workspace_graph_id);
create index if not exists idx_runs_task_graph on runs(task_graph_id);
create index if not exists idx_nodes_task_graph on nodes(task_graph_id);
create index if not exists idx_task_edges_task_graph on task_edges(task_graph_id);
create index if not exists idx_workspace_edges_from on workspace_edges(from_type, from_id);
create index if not exists idx_workspace_edges_to on workspace_edges(to_type, to_id);
create index if not exists idx_run_node_states_status on run_node_states(run_id, status);
create index if not exists idx_artifacts_run on artifacts(run_id);
create index if not exists idx_embeddings_owner on embeddings(owner_type, owner_id);
