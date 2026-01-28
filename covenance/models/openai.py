from enum import Enum

from covenance.pricing import ModelPricing


class OpenAIModels(str, Enum):
    # Reasoning models
    o3 = "o3"
    o3pro = "o3-pro"
    o3mini = "o3-mini"
    o4mini = "o4-mini"
    o1 = "o1"
    o1mini = "o1-mini"

    # GPT-5 family
    gpt5pro = "gpt-5-pro"
    gpt5 = "gpt-5"
    gpt5mini = "gpt-5-mini"
    gpt5nano = "gpt-5-nano"
    gpt51 = "gpt-5.1"
    gpt52 = "gpt-5.2"

    # GPT-4.1 family
    gpt41 = "gpt-4.1"
    gpt41mini = "gpt-4.1-mini"
    gpt41nano = "gpt-4.1-nano"

    # GPT-4o family
    gpt4o = "gpt-4o"
    gpt4o_mini = "gpt-4o-mini"


# Pricing per 1M tokens (verified 2026-01-27, Standard tier)
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
    "o3-pro": ModelPricing(input=20.00, output=80.00, cached=None),
    "o4-mini": ModelPricing(input=1.10, output=4.40, cached=0.275),
    "o1": ModelPricing(input=15.00, output=60.00, cached=7.50),
    "o1-mini": ModelPricing(input=1.10, output=4.40, cached=0.55),
    "o3-mini": ModelPricing(input=1.10, output=4.40, cached=0.55),
}
