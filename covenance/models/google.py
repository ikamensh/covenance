from enum import Enum

from covenance.pricing import ModelPricing


class GeminiModels(str, Enum):
    # Gemini 3 preview
    pro_3 = "gemini-3-pro-preview"
    flash_3 = "gemini-3-flash-preview"

    # Gemini 2.5 - current "stable" generation as of July 2025
    pro_25 = "gemini-2.5-pro"  # best reasoning/coding
    flash_25 = "gemini-2.5-flash"  # fast / inexpensive
    flash_lite_25 = "gemini-2.5-flash-lite"  # extra-cheap, small context

    # Gemini 2.0
    flash_20 = "gemini-2.0-flash"
    flash_lite_20 = "gemini-2.0-flash-lite"


# Pricing per 1M tokens (verified 2026-01-27, Standard tier, prompts <= 200k)
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
