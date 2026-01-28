from enum import Enum

from covenance.pricing import ModelPricing


class ClaudeModels(str, Enum):
    """Anthropic Claude model identifiers.

    See: https://docs.anthropic.com/claude/docs/models-overview
    """

    opus45 = "claude-opus-4-5"
    sonnet45 = "claude-sonnet-4-5"
    haiku45 = "claude-haiku-4-5"


# Pricing per 1M tokens (from LiteLLM 2026-01-27)
# Cache discount is 90% (cached = 10% of input)
CLAUDE_PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-5": ModelPricing(input=5.00, output=25.00, cached=0.50),
    "claude-sonnet-4-5": ModelPricing(input=3.00, output=15.00, cached=0.30),
    "claude-haiku-4-5": ModelPricing(input=1.00, output=5.00, cached=0.10),
}
