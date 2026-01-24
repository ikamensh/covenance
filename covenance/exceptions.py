"""Exceptions for LLM API clients."""


class StructuredOutputParsingError(ValueError):
    """Exception raised when LLM API returns a response but parsed field is None.

    This typically indicates a schema mismatch or parsing error. The unified
    wrapper will retry these errors automatically.
    """

    pass
