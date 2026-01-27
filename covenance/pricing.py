"""Model pricing data for cost calculation.

Prices are per 1M tokens (standard tier, text).
Sources:
  OpenAI: https://platform.openai.com/docs/pricing
  Gemini: https://ai.google.dev/gemini-api/docs/pricing

Last verified: 2026-01-27

Note: cached_tokens is a subset of input_tokens in both APIs.
Cost = (input_tokens - cached_tokens) * input_price + cached_tokens * cached_price + output_tokens * output_price
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


# OpenAI pricing (verified 2026-01-27, Standard tier)
# Cache discounts vary by model family:
#   gpt-5.x: 90% off (cached = 10% of input)
#   gpt-4.1: 75% off (cached = 25% of input)
#   gpt-4o: 50% off (cached = 50% of input)
#   o3/o4-mini: 75% off (cached = 25% of input)
OPENAI_PRICING: dict[str, ModelPricing] = {
    # GPT-5 family (90% cache discount)
    "gpt-5.2": ModelPricing(input=1.75, output=14.00, cached=0.175),
    "gpt-5.1": ModelPricing(input=1.25, output=10.00, cached=0.125),
    "gpt-5": ModelPricing(input=1.25, output=10.00, cached=0.125),
    "gpt-5-mini": ModelPricing(input=0.25, output=2.00, cached=0.025),
    "gpt-5-nano": ModelPricing(input=0.05, output=0.40, cached=0.005),
    # GPT-5 aliases
    "gpt-5.2-chat-latest": ModelPricing(input=1.75, output=14.00, cached=0.175),
    "gpt-5.1-chat-latest": ModelPricing(input=1.25, output=10.00, cached=0.125),
    "gpt-5-chat-latest": ModelPricing(input=1.25, output=10.00, cached=0.125),
    # GPT-4.1 family (75% cache discount)
    "gpt-4.1": ModelPricing(input=2.00, output=8.00, cached=0.50),
    "gpt-4.1-mini": ModelPricing(input=0.40, output=1.60, cached=0.10),
    "gpt-4.1-nano": ModelPricing(input=0.10, output=0.40, cached=0.025),
    # GPT-4o family (50% cache discount)
    "gpt-4o": ModelPricing(input=2.50, output=10.00, cached=1.25),
    "gpt-4o-mini": ModelPricing(input=0.15, output=0.60, cached=0.075),
    # Reasoning models
    "o3": ModelPricing(input=2.00, output=8.00, cached=0.50),
    "o4-mini": ModelPricing(input=1.10, output=4.40, cached=0.275),
    "o1": ModelPricing(input=15.00, output=60.00, cached=7.50),
    "o1-mini": ModelPricing(input=1.10, output=4.40, cached=0.55),
    "o3-mini": ModelPricing(input=1.10, output=4.40, cached=0.55),
}

# Gemini pricing (verified 2026-01-27, Standard tier, prompts <= 200k)
# Cache discount is 90% for 2.5+ models (cached = 10% of input)
# Cache discount is 75% for 2.0 models (cached = 25% of input)
GEMINI_PRICING: dict[str, ModelPricing] = {
    # Gemini 3 preview (90% cache discount)
    "gemini-3-pro-preview": ModelPricing(input=2.00, output=12.00, cached=0.20),
    "gemini-3-flash-preview": ModelPricing(input=0.50, output=3.00, cached=0.05),
    # Gemini 2.5 stable (90% cache discount)
    "gemini-2.5-pro": ModelPricing(input=1.25, output=10.00, cached=0.125),
    "gemini-2.5-flash": ModelPricing(input=0.30, output=2.50, cached=0.03),
    "gemini-2.5-flash-lite": ModelPricing(input=0.10, output=0.40, cached=0.01),
    # Gemini 2.0 (75% cache discount)
    "gemini-2.0-flash": ModelPricing(input=0.10, output=0.40, cached=0.025),
    "gemini-2.0-flash-lite": ModelPricing(
        input=0.075, output=0.30, cached=None
    ),  # no caching
}

# xAI Grok pricing (verified 2026-01-27)
# Sources: https://docs.x.ai/docs/models, https://pricepertoken.com/pricing-page/provider/xai
# Note: xAI uses dashes in model names (grok-4-1-fast, not grok-4.1-fast)
# Cache discount is 75% (cached = 25% of input)
GROK_PRICING: dict[str, ModelPricing] = {
    # Flagship reasoning model (256k context)
    "grok-4": ModelPricing(input=3.00, output=15.00, cached=0.75),
    # Fast models (2M context) - reasoning enabled by default
    "grok-4-1-fast": ModelPricing(input=0.20, output=0.50, cached=0.05),
    "grok-4-fast": ModelPricing(input=0.20, output=0.50, cached=0.05),
    # Non-reasoning variants (faster, no chain-of-thought)
    "grok-4-1-fast-non-reasoning": ModelPricing(input=0.20, output=0.50, cached=0.05),
    "grok-4-fast-non-reasoning": ModelPricing(input=0.20, output=0.50, cached=0.05),
    # Code-specialized (256k context, 90% cache discount)
    "grok-code-fast-1": ModelPricing(input=0.20, output=1.50, cached=0.02),
    # Grok 3 family (131k context)
    "grok-3": ModelPricing(input=3.00, output=15.00, cached=0.75),
    "grok-3-mini": ModelPricing(input=0.30, output=0.50, cached=0.075),
}

# Combined lookup by provider
PRICING: dict[str, dict[str, ModelPricing]] = {
    "openai": OPENAI_PRICING,
    "gemini": GEMINI_PRICING,
    "grok": GROK_PRICING,
}


def get_model_pricing(model: str, provider: str) -> ModelPricing | None:
    """Get pricing for a model, or None if unknown."""
    provider_pricing = PRICING.get(provider)
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
