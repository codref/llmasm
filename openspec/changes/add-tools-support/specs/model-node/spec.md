## ADDED Requirements

### Requirement: Model nodes support tool-use execution mode
A model node SHALL be able to execute in a tool-use mode where the LLM may request tool invocations and receive results before producing a final output.

#### Scenario: Tool-use enabled model node
- **WHEN** a model node's execution configuration enables tools
- **THEN** the executor enters a tool-use loop instead of returning a single response

#### Scenario: Tool-use disabled model node
- **WHEN** a model node's execution configuration does not enable tools
- **THEN** the executor performs a single generation as it does today

### Requirement: Model node tool-use output is coerced to the output schema
After any tool-use rounds complete, the final model response SHALL be coerced to the node's configured output schema.

#### Scenario: Final answer after tool use
- **WHEN** a model node with output schema `FinalAnswer` completes its tool-use loop
- **THEN** the returned artifact is a valid `FinalAnswer`

#### Scenario: Raw text after tool use
- **WHEN** a model node with output schema `RawText` completes its tool-use loop
- **THEN** the returned artifact is a valid `RawText`
