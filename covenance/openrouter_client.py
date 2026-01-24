"""OpenRouter client with structured output support and automatic retry.

OpenRouter provides access to multiple LLM providers through a unified API.
It's OpenAI-compatible, so we use the OpenAI SDK with OpenRouter's base URL.
"""

from enum import Enum
from typing import TypeVar

from openai import OpenAI

from ._lazy_client import LazyClient
from .keys import get_openrouter_api_key, require_api_key
from .openai_client import ask_openai_compatible_structured

T = TypeVar("T")

# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _create_openrouter_client() -> OpenAI:
    api_key = require_api_key(
        get_openrouter_api_key(), "openrouter", ["OPENROUTER_API_KEY"]
    )
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


client = LazyClient(_create_openrouter_client, label="openrouter")


class OpenRouterModels(str, Enum):
    """OpenRouter model identifiers.

    OpenRouter provides access to models from multiple providers.
    Format: provider/model-name

    See: https://openrouter.ai/models for full list
    """

    # Meta models via OpenRouter
    llama_3_1_70b = "meta-llama/llama-3.1-70b-instruct"
    llama_3_1_8b = "meta-llama/llama-3.1-8b-instruct"

    nova2_lite_free = "amazon/nova-2-lite-v1:free"
    deepseek32 = "deepseek/deepseek-v3.2"
    deepseek32_high = "deepseek/deepseek-v3.2-speciale"
    gpt_oss_120b = "openai/gpt-oss-120b"


def ask_openrouter_structured[T](
    user_msg: str,
    format: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = OpenRouterModels.nova2_lite_free,
) -> T:
    """Call OpenRouter API with automatic retry."""
    return ask_openai_compatible_structured(
        client=client,
        user_msg=user_msg,
        format=format,
        sys_msg=sys_msg,
        model=model,
        provider="openrouter",
    )


if __name__ == "__main__":
    from pydantic import BaseModel

    class MovieReview(BaseModel):
        movie_title: str
        sentiment: str
        rating: float
        key_themes: list[str]

    result = ask_openrouter_structured(
        user_msg="Review the movie 'Inception' by Christopher Nolan.",
        format=MovieReview,
        model=OpenRouterModels.deepseek32,
    )

    print(f"Result: {result.model_dump_json(indent=4)}")
