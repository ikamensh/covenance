"""Exceptions for LLM API clients."""


class StructuredOutputParsingError(ValueError):
    """Exception raised when LLM API returns a response but parsed field is None.

    This typically indicates a schema mismatch or parsing error. The unified
    wrapper will retry these errors automatically.
    """

    pass


class MissingProviderError(ImportError):
    """Raised when a provider SDK is not installed."""

    pass


def require_provider(provider: str) -> None:
    """Import check with helpful error message for optional provider dependencies.

    Args:
        provider: One of 'openai', 'anthropic', 'google', 'mistral'
    """
    extra = provider
    package_map = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google-genai",
        "mistral": "mistralai",
    }
    package = package_map.get(provider, provider)
    try:
        if provider == "openai":
            import openai  # noqa: F401
        elif provider == "anthropic":
            import anthropic  # noqa: F401
        elif provider == "google":
            import google.genai  # noqa: F401
        elif provider == "mistral":
            import mistralai  # noqa: F401
    except ImportError as e:
        raise MissingProviderError(
            f"The '{package}' package is required for {provider} models. "
            f"Install it with: pip install covenance[{extra}]"
        ) from e
