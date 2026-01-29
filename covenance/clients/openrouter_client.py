"""OpenRouter client with structured output support and automatic retry.

OpenRouter provides access to multiple LLM providers through a unified API.
It's OpenAI-compatible, so we use the OpenAI SDK with OpenRouter's base URL.
"""

from typing import TYPE_CHECKING, TypeVar

from covenance._lazy_client import LazyClient
from covenance.exceptions import require_provider
from covenance.keys import get_openrouter_api_key, require_api_key
from covenance.models import OpenRouterModels

from .openai_client import ask_openai_compatible_structured

if TYPE_CHECKING:
    from openai import OpenAI

    from covenance.record import RecordStore

T = TypeVar("T")

# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _create_openrouter_client() -> "OpenAI":
    require_provider("openai")
    from openai import OpenAI

    api_key = require_api_key(get_openrouter_api_key(), "openrouter")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


client = LazyClient(_create_openrouter_client, label="openrouter")


def ask_openrouter[T](
    user_msg: str,
    response_type: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = OpenRouterModels.nova2_lite_free,
    *,
    client_override: "OpenAI | None" = None,
    record_store: "RecordStore | None" = None,
    temperature: float | None = None,
) -> T:
    """Call OpenRouter API with automatic retry."""
    api_client = client_override or client  # type: ignore[assignment]
    return ask_openai_compatible_structured(
        client=api_client,
        user_msg=user_msg,
        response_type=response_type,
        sys_msg=sys_msg,
        model=model,
        provider="openrouter",
        record_store=record_store,
        temperature=temperature,
    )


if __name__ == "__main__":
    from pydantic import BaseModel

    class MovieReview(BaseModel):
        movie_title: str
        sentiment: str
        rating: float
        key_themes: list[str]

    result = ask_openrouter(
        user_msg="Review the movie 'Inception' by Christopher Nolan.",
        response_type=MovieReview,
        model=OpenRouterModels.deepseek32,
    )

    print(f"Result: {result.model_dump_json(indent=4)}")
