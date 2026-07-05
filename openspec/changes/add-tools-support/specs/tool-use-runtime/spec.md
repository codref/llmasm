## ADDED Requirements

### Requirement: Model nodes can advertise available tools
A model node SHALL advertise zero or more registered tools to the LLM based on its execution configuration.

#### Scenario: Default tool availability
- **WHEN** a model node has no `tools` execution key
- **THEN** no tools are advertised to the model

#### Scenario: Advertise all tools
- **WHEN** a model node's `tools` execution key is `"all"`
- **THEN** every registered tool is advertised as a JSON Schema function definition

#### Scenario: Advertise selected tools
- **WHEN** a model node's `tools` execution key is a list of tool names
- **THEN** only those tools are advertised

### Requirement: Executor invokes tool calls requested by the model
When a model response contains tool calls, the executor SHALL invoke the corresponding registered tools and feed the results back to the model.

#### Scenario: Single tool call
- **WHEN** the model requests one tool call
- **THEN** the executor invokes that tool once and stores the result

#### Scenario: Parallel tool calls
- **WHEN** the model requests multiple tool calls in one response
- **THEN** the executor invokes all tools and returns all results in the same follow-up

#### Scenario: Unknown tool call
- **WHEN** the model requests a tool that is not registered
- **THEN** the executor records an error result and feeds it back without raising

### Requirement: Tool-use loop is bounded
The executor SHALL cap the number of tool-use rounds to prevent infinite loops.

#### Scenario: Model keeps requesting tools
- **WHEN** the model requests tools in every round up to the configured maximum
- **THEN** after the maximum is reached the node succeeds with the last tool results as text

### Requirement: Tool-use artifacts are persisted
Each tool-use round SHALL produce artifacts for the tool calls and tool results.

#### Scenario: Successful round produces artifacts
- **WHEN** a model node uses tools
- **THEN** `tool_calls` and `tool_results` artifacts are persisted on the node
