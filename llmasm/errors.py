"""Exception hierarchy for LLMASM."""


class LLMASMError(Exception):
    """Base class for all LLMASM exceptions."""


class ValidationError(LLMASMError):
    """Raised when a hard validation boundary is violated."""


class CompilationError(LLMASMError):
    """Raised when compilation cannot produce a valid task graph."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int | None = None,
        last_errors: object | None = None,
        last_raw_output: str | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_errors = last_errors
        self.last_raw_output = last_raw_output


class ExecutionError(LLMASMError):
    """Base class for execution-time failures."""


class RetryableError(ExecutionError):
    """Raised by node handlers when a transient failure should be retried."""


class FatalError(ExecutionError):
    """Raised by node handlers when a run must fail immediately."""


class StorageError(LLMASMError):
    """Raised for storage failures."""


class ProviderError(LLMASMError):
    """Raised for provider failures."""
