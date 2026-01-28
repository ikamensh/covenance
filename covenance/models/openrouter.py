from enum import Enum


class OpenRouterModels(str, Enum):
    """OpenRouter model identifiers.

    OpenRouter provides access to models from multiple providers.
    Format: provider/model-name

    See: https://openrouter.ai/models for full list

    Note: OpenRouter pricing varies and is pass-through from providers.
    We don't track pricing for OpenRouter models.
    """

    # Meta models via OpenRouter
    llama_31_70b = "meta-llama/llama-3.1-70b-instruct"
    llama_31_8b = "meta-llama/llama-3.1-8b-instruct"

    nova2_lite_free = "amazon/nova-2-lite-v1:free"
    deepseek32 = "deepseek/deepseek-v3.2"
    deepseek32_high = "deepseek/deepseek-v3.2-speciale"
    gpt_oss_120b = "openai/gpt-oss-120b"


# OpenRouter pricing is pass-through; no local tracking
OPENROUTER_PRICING: dict = {}
