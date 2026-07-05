## ADDED Requirements

### Requirement: Tool registry emits JSON Schema definitions
The tool registry SHALL be able to convert any registered tool's input schema into a JSON Schema function definition suitable for an LLM.

#### Scenario: Convert a Pydantic input schema
- **WHEN** a tool's input schema is a registered Pydantic model
- **THEN** the registry returns a JSON Schema object describing the model's fields

#### Scenario: Unknown tool returns an error
- **WHEN** the registry is asked for a tool that is not registered
- **THEN** it raises `ValidationError`

### Requirement: Built-in tools.list introspection tool
A built-in `tools.list` tool SHALL be available that returns the names and descriptions of registered tools.

#### Scenario: List registered tools
- **WHEN** `tools.list` is invoked
- **THEN** it returns a structured list containing each registered tool's name and description

#### Scenario: tools.list is registered by default
- **WHEN** a `ToolRegistry` is created
- **THEN** `tools.list` is present without manual registration

### Requirement: Tool spec helpers accept typed Pydantic models
Tool authors SHALL be able to define a tool by providing a Pydantic input model and a callable, without manually writing a JSON Schema.

#### Scenario: Decorator-based tool definition
- **WHEN** a developer defines a function annotated with a Pydantic input model
- **THEN** a `ToolSpec` and `Tool` implementation are generated automatically
