from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypeVar

from covenance._lazy_client import LazyClient
from covenance.exceptions import StructuredOutputParsingError, require_provider
from covenance.keys import get_openai_api_key, require_api_key
from covenance.models import OpenAIModels

if TYPE_CHECKING:
    from openai import OpenAI

    from covenance.record import RecordStore, TokenUsage

T = TypeVar("T")


def _create_openai_client() -> OpenAI:
    require_provider("openai")
    from openai import OpenAI

    api_key = require_api_key(get_openai_api_key(), "openai")
    return OpenAI(api_key=api_key)


client = LazyClient(_create_openai_client, label="openai")

# Global verbose flag for retry logging
VERBOSE = False


def _parse_wait_time_from_error(error: Exception) -> float:
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
    from covenance.record import TokenUsage

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

    return usage


def ask_openai_compatible_structured[T](
    client: OpenAI,
    user_msg: str,
    response_type: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = "gpt-4o",
    provider: str = "openai",
    record_store: RecordStore | None = None,
    temperature: float | None = None,
) -> T:
    """Execute structured call against an OpenAI-compatible API with retries."""
    from openai import RateLimitError

    max_attempts = 100
    total_tpm_wait = 0.0
    started_at = datetime.now(UTC)
    is_plain_text = response_type is str or response_type is None
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
                    temperature=temperature,
                )
                output = response.output_text
            else:
                response = client.responses.parse(
                    model=model,
                    input=user_msg,
                    text_format=response_type,
                    instructions=sys_msg,
                    temperature=temperature,
                )
                output = response.output_parsed

            ended_at = datetime.now(UTC)
            usage = _extract_openai_compatible_usage(
                response, model=model, provider=provider
            )

            from covenance.record import record_llm_call

            record_llm_call(
                model=model,
                provider=provider,
                usage=usage,
                tpm_retry_wait_seconds=total_tpm_wait,
                started_at=started_at,
                ended_at=ended_at,
                record_store=record_store,
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


def ask_openai[T](
    user_msg: str,
    response_type: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = OpenAIModels.gpt5.value,
    *,
    client_override: OpenAI | None = None,
    record_store: RecordStore | None = None,
    temperature: float | None = None,
) -> T:
    """Call OpenAI API with automatic retry."""
    api_client = client_override or client  # type: ignore[assignment]
    return ask_openai_compatible_structured(
        client=api_client,
        user_msg=user_msg,
        response_type=response_type,
        sys_msg=sys_msg,
        model=model,
        provider="openai",
        record_store=record_store,
        temperature=temperature,
    )


if __name__ == "__main__":
    from pydantic import BaseModel

    class Response(BaseModel):
        text: str
        number: int

    out = ask_openai(
        "What is the capital?",
        response_type=Response,
        model="o4-mini",
        # sys_msg="You are guessy assistant. Guess any missing information. Never ask for any clarifications."
    )
    print(f"Result: {out}")
