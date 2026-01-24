import re
import time
from datetime import UTC, datetime
from enum import Enum
from typing import TypeVar

from openai import OpenAI, RateLimitError

from ._lazy_client import LazyClient
from .exceptions import StructuredOutputParsingError
from .keys import get_openai_api_key, require_api_key
from .usage import TokenUsage

T = TypeVar("T")


class OpenaiModels(str, Enum):
    o3 = "o3"
    o3pro = "o3-pro"

    gpt5pro = "gpt-5-pro"
    gpt5 = "gpt-5"
    gpt5mini = "gpt-5-mini"
    gpt_5_nano = "gpt-5-nano"
    gpt51 = "gpt-5.1"
    gpt52 = "gpt-5.2"


def _create_openai_client() -> OpenAI:
    api_key = require_api_key(get_openai_api_key(), "openai", ["OPENAI_API_KEY"])
    return OpenAI(api_key=api_key)


client = LazyClient(_create_openai_client, label="openai")

# Global verbose flag for retry logging
VERBOSE = False


def _parse_wait_time_from_error(error: RateLimitError) -> float:
    """Parse wait time from OpenAI RateLimitError message.

    The error message typically contains: "Please try again in X.XXXs"

    Args:
        error: The RateLimitError exception

    Returns:
        Wait time in seconds, or 1.0 if parsing fails
    """
    error_message = str(error)
    # Look for pattern like "Please try again in 6.191s" or "Please try again in 6s"
    # Match both integer and decimal numbers (e.g., "6.191", "6", "0.5")
    match = re.search(r"Please try again in ([0-9]+(?:\.[0-9]+)?)s", error_message)
    if match:
        try:
            wait_time = float(match.group(1))
            # Ensure we wait at least a small amount
            return max(wait_time, 0.1)
        except ValueError:
            pass
    # Fallback: use a default wait time
    return 1.0


def set_rate_limiter_verbose(verbose: bool) -> None:
    """Enable or disable verbose logging for OpenAI retry logic.

    Args:
        verbose: If True, print detailed logging about retry attempts and wait times
    """
    global VERBOSE
    VERBOSE = verbose


def _extract_openai_compatible_usage(
    response, model: str, provider: str = "openai"
) -> TokenUsage:
    """Extract token usage from OpenAI-compatible response."""
    if not hasattr(response, "usage") or response.usage is None:
        p_name = "OpenAI" if provider == "openai" else provider.capitalize()
        raise AttributeError(f"{p_name} response missing usage info for {model}")

    u = response.usage
    prompt_tokens = u.input_tokens
    completion_tokens = u.output_tokens
    cached_tokens = 0

    if hasattr(u, "input_tokens_details") and u.input_tokens_details:
        cached_tokens = getattr(u.input_tokens_details, "cached_tokens", 0)

    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=u.total_tokens,
        cached_tokens=cached_tokens or 0,
    )

    from .usage import usage_stats

    usage_stats.record_usage(usage, model=model, provider=provider)
    return usage


def ask_openai_compatible_structured[T](
    client: OpenAI,
    user_msg: str,
    format: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = "gpt-4o",
    provider: str = "openai",
) -> T:
    """Execute structured call against an OpenAI-compatible API with retries."""
    max_attempts = 100
    total_tpm_wait = 0.0
    started_at = datetime.now(UTC)
    is_plain_text = format is str or format is None

    for attempt in range(max_attempts):
        try:
            if VERBOSE and attempt > 0:
                print(
                    f"[{provider.capitalize()} Retry] Attempt {attempt + 1} for {model}"
                )

            if is_plain_text:
                response = client.responses.create(
                    model=model,
                    input=user_msg,
                    instructions=sys_msg,
                )
                output = response.output_text
            else:
                response = client.responses.parse(
                    model=model,
                    input=user_msg,
                    text_format=format,
                    instructions=sys_msg,
                )
                output = response.output_parsed

            ended_at = datetime.now(UTC)
            usage = _extract_openai_compatible_usage(
                response, model=model, provider=provider
            )

            from .metrics import record_llm_call

            record_llm_call(
                model=model,
                provider=provider,
                usage=usage,
                tpm_retry_wait_seconds=total_tpm_wait,
                started_at=started_at,
                ended_at=ended_at,
            )

            if output is None:
                raise StructuredOutputParsingError(
                    f"Empty output from {provider}/{model}"
                )

            return output  # type: ignore[return-value]

        except RateLimitError as e:
            if attempt == max_attempts - 1:
                raise
            wait_time = max(_parse_wait_time_from_error(e), 1.0)
            if VERBOSE:
                print(
                    f"[{provider.capitalize()} Retry] Rate limit, waiting {wait_time:.2f}s"
                )
            time.sleep(wait_time)
            total_tpm_wait += wait_time


def ask_chatgpt_structured[T](
    user_msg: str,
    format: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = OpenaiModels.gpt5.value,
) -> T:
    """Call OpenAI API with automatic retry."""
    return ask_openai_compatible_structured(
        client=client,
        user_msg=user_msg,
        format=format,
        sys_msg=sys_msg,
        model=model,
        provider="openai",
    )


if __name__ == "__main__":
    from pydantic import BaseModel

    class Response(BaseModel):
        text: str
        number: int

    out = ask_chatgpt_structured(
        "What is the capital?",
        format=Response,
        model="o4-mini",
        # sys_msg="You are guessy assistant. Guess any missing information. Never ask for any clarifications."
    )
    print(f"Result: {out}")
