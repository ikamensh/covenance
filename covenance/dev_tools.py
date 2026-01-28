from __future__ import annotations


def set_rate_limiter_verbose(verbose: bool) -> None:
    """Enable or disable verbose logging for all rate limiters."""
    from .clients.anthropic_client import (
        set_rate_limiter_verbose as set_anthropic_verbose,
    )
    from .clients.google_client import set_rate_limiter_verbose as set_gemini_verbose
    from .clients.mistral_client import set_rate_limiter_verbose as set_mistral_verbose
    from .clients.openai_client import set_rate_limiter_verbose as set_openai_verbose

    set_anthropic_verbose(verbose)
    set_gemini_verbose(verbose)
    set_openai_verbose(verbose)
    set_mistral_verbose(verbose)
