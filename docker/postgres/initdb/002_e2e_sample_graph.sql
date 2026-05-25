LOAD 'age';
SET search_path = ag_catalog, "$user", public;

SELECT *
FROM cypher('llmasm_graph', $$
  CREATE
    (intent:TaskNode {
      id: 'node_intent',
      kind: 'intent',
      name: 'intent',
      schema: 'RawText',
      status: 'succeeded'
    }),
    (docs:TaskNode {
      id: 'node_retrieve_incident_docs',
      kind: 'tool',
      name: 'retrieve_incident_docs',
      tool: 'corpus.search_incident_docs',
      schema: 'RawText',
      status: 'succeeded'
    }),
    (tickets:TaskNode {
      id: 'node_retrieve_tickets',
      kind: 'tool',
      name: 'retrieve_tickets',
      tool: 'corpus.search_tickets',
      schema: 'RawText',
      status: 'succeeded'
    }),
    (facts:TaskNode {
      id: 'node_extract_incident_facts',
      kind: 'model',
      name: 'extract_incident_facts',
      model: 'gemma4:e4b',
      schema: 'Summary',
      status: 'succeeded'
    }),
    (risks:TaskNode {
      id: 'node_extract_ticket_risks',
      kind: 'model',
      name: 'extract_ticket_risks',
      model: 'gemma4:e4b',
      schema: 'Summary',
      status: 'succeeded'
    }),
    (brief:TaskNode {
      id: 'node_synthesize_execution_brief',
      kind: 'model',
      name: 'synthesize_execution_brief',
      model: 'gemma4:e4b',
      schema: 'Summary',
      status: 'succeeded'
    }),
    (final:TaskNode {
      id: 'node_final',
      kind: 'final',
      name: 'final',
      schema: 'FinalAnswer',
      status: 'succeeded'
    }),
    (goal:Goal {
      id: 'goal_complex_e2e',
      text: 'Create an incident-informed Q3 checkout execution brief.'
    }),
    (task:TaskGraph {
      id: 'taskgraph_complex_e2e',
      intent: 'produce incident-informed Q3 execution brief'
    }),
    (task)-[:CONTAINS]->(intent),
    (task)-[:CONTAINS]->(docs),
    (task)-[:CONTAINS]->(tickets),
    (task)-[:CONTAINS]->(facts),
    (task)-[:CONTAINS]->(risks),
    (task)-[:CONTAINS]->(brief),
    (task)-[:CONTAINS]->(final),
    (task)-[:SUPPORTS_GOAL]->(goal),
    (intent)-[:DATAFLOW {from_port: 'output', to_port: 'input'}]->(docs),
    (intent)-[:DATAFLOW {from_port: 'output', to_port: 'input'}]->(tickets),
    (docs)-[:DATAFLOW {from_port: 'output', to_port: 'incident_docs'}]->(facts),
    (tickets)-[:DATAFLOW {from_port: 'output', to_port: 'tickets'}]->(risks),
    (facts)-[:DATAFLOW {from_port: 'output', to_port: 'incident_facts'}]->(brief),
    (risks)-[:DATAFLOW {from_port: 'output', to_port: 'ticket_risks'}]->(brief),
    (brief)-[:DATAFLOW {from_port: 'output', to_port: 'input'}]->(final)
  RETURN task
$$) AS (task agtype);
