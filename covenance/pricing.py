"""Model pricing utilities.

Pricing data lives alongside model enums in covenance/models/*.py.
This module provides the ModelPricing dataclass and cost calculation functions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a model in USD per 1M tokens.

    Attributes:
        input: Price per 1M input tokens (fresh, non-cached)
        output: Price per 1M output tokens
        cached: Price per 1M cached input tokens (None if caching not supported)
    """

    input: float
    output: float
    cached: float | None = None


def _get_pricing() -> dict[str, dict[str, ModelPricing]]:
    """Load pricing dicts from model files."""
    from covenance.models.anthropic import CLAUDE_PRICING
    from covenance.models.google import GEMINI_PRICING
    from covenance.models.grok import GROK_PRICING
    from covenance.models.openai import OPENAI_PRICING

    return {
        "openai": OPENAI_PRICING,
        "gemini": GEMINI_PRICING,
        "grok": GROK_PRICING,
        "anthropic": CLAUDE_PRICING,
    }


def get_model_pricing(model: str, provider: str) -> ModelPricing | None:
    """Get pricing for a model, or None if unknown."""
    pricing = _get_pricing()
    provider_pricing = pricing.get(provider)
    if provider_pricing is None:
        return None
    return provider_pricing.get(model)


def calculate_cost(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float | None:
    """Calculate cost in USD, or None if pricing unknown.

    cached_tokens is a subset of input_tokens (i.e., cached_tokens <= input_tokens).
    Fresh input tokens = input_tokens - cached_tokens.
    """
    pricing = get_model_pricing(model, provider)
    if pricing is None:
        return None

    fresh_input = input_tokens - cached_tokens
    fresh_input_cost = (fresh_input / 1_000_000) * pricing.input

    # If model supports caching, use cached price; otherwise treat as full price
    cached_price = pricing.cached if pricing.cached is not None else pricing.input
    cached_cost = (cached_tokens / 1_000_000) * cached_price

    output_cost = (output_tokens / 1_000_000) * pricing.output

    return round(fresh_input_cost + cached_cost + output_cost, 6)
