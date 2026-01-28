from enum import Enum

from covenance.pricing import ModelPricing


class MistralModels(str, Enum):
    """Mistral AI model identifiers.

    See: https://docs.mistral.ai/getting-started/models/models_overview/
    """

    # Flagship models
    large = "mistral-large-latest"
    medium = "mistral-medium-latest"
    small = "mistral-small-latest"

    # Edge models
    ministral_8b = "ministral-8b-latest"
    ministral_3b = "ministral-3b-latest"

    # Specialized
    codestral = "codestral-latest"


# Pricing per 1M tokens (from LiteLLM 2026-01-27)
# Mistral doesn't support caching
MISTRAL_PRICING: dict[str, ModelPricing] = {
    "mistral-large-latest": ModelPricing(input=2.00, output=6.00, cached=None),
    "mistral-medium-latest": ModelPricing(input=0.40, output=2.00, cached=None),
    "mistral-small-latest": ModelPricing(input=0.10, output=0.30, cached=None),
    "codestral-latest": ModelPricing(input=1.00, output=3.00, cached=None),
}
