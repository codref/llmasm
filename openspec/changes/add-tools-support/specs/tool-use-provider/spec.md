## ADDED Requirements

### Requirement: Provider accepts tool definitions
The LLM provider protocol SHALL allow callers to pass a list of tool definitions to the generate method.

#### Scenario: Tool definitions are optional
- **WHEN** a caller invokes `generate` without tools
- **THEN** the provider behaves exactly as before and does not require a tools argument

#### Scenario: Tool definitions are forwarded to the model
- **WHEN** a caller invokes `generate` with a non-empty `tools` list
- **THEN** the provider includes those tool definitions in the request to the model

### Requirement: Ollama provider uses chat endpoint for tool calls
The Ollama provider SHALL use the `/api/chat` endpoint when tools are provided, and `/api/generate` otherwise.

#### Scenario: No tools uses generate endpoint
- **WHEN** `generate` is called with no tools
- **THEN** the request is sent to `/api/generate`

#### Scenario: Tools present uses chat endpoint
- **WHEN** `generate` is called with at least one tool
- **THEN** the request is sent to `/api/chat` with a messages array and a tools array

### Requirement: Tool calls are parsed from model response
The Ollama provider SHALL parse `tool_calls` from the chat response and return them in `ModelOutput.tool_calls`.

#### Scenario: Model returns a plain text response
- **WHEN** the model response contains no `tool_calls`
- **THEN** `tool_calls` is empty and `text` contains the response content

#### Scenario: Model returns tool calls
- **WHEN** the model response contains one or more `tool_calls`
- **THEN** `tool_calls` contains the parsed calls with name and arguments

### Requirement: Tool results can be sent back to the model
The Ollama provider SHALL accept a follow-up messages list containing previous assistant tool calls and user tool results.

#### Scenario: Round-trip tool result
- **WHEN** the provider is called with messages that include a `tool` role message
- **THEN** the request includes that message and the model can produce a follow-up response
