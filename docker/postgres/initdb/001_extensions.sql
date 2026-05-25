CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;

LOAD 'age';
SET search_path = ag_catalog, "$user", public;

SELECT create_graph('llmasm_graph')
WHERE NOT EXISTS (
  SELECT 1
  FROM ag_catalog.ag_graph
  WHERE name = 'llmasm_graph'
);
