from enum import Enum

from covenance.pricing import ModelPricing


class GrokModels(str, Enum):
    """xAI Grok model identifiers.

    Note: xAI uses dashes not dots in model names (grok-4-1-fast, not grok-4.1-fast).
    "non-reasoning" variants disable chain-of-thought for faster responses.
    """

    grok4 = "grok-4"
    grok41_fast = "grok-4-1-fast"
    grok41_fast_nonreasoning = "grok-4-1-fast-non-reasoning"
    grok4_fast = "grok-4-fast"
    grok4_fast_nonreasoning = "grok-4-fast-non-reasoning"
    grok_code_fast = "grok-code-fast-1"
    grok3 = "grok-3"
    grok3_mini = "grok-3-mini"


# Pricing per 1M tokens (verified 2026-01-27)
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
