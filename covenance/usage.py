"""Token usage types."""

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Standardized token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0  # Tokens read from cache (provider-specific support)
