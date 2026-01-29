"""Anthropic Claude client with structured output support and automatic retry.

Uses the structured outputs beta (constrained decoding) when SDK >= 0.74.1,
providing guaranteed schema-valid JSON. Falls back to tool-use for older SDKs.
"""

import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypeVar

from covenance._lazy_client import LazyClient
from covenance.exceptions import StructuredOutputParsingError, require_provider
from covenance.keys import get_anthropic_api_key, require_api_key
from covenance.models import ClaudeModels
from covenance.record import TokenUsage
from covenance.retry import exponential_backoff

if TYPE_CHECKING:
    from anthropic import Anthropic

    from covenance.record import RecordStore

T = TypeVar("T")

# Check SDK version for structured outputs beta support (requires >= 0.74.1)
_USE_STRUCTURED_OUTPUTS_BETA = False
try:
    from anthropic import __version__ as _anthropic_version

    _major, _minor, _patch = map(int, _anthropic_version.split(".")[:3])
    _USE_STRUCTURED_OUTPUTS_BETA = (_major, _minor, _patch) >= (0, 74, 1)
except Exception:
    pass  # Fall back to tool-use if version check fails


def _create_anthropic_client() -> "Anthropic":
    require_provider("anthropic")
    from anthropic import Anthropic

    api_key = require_api_key(get_anthropic_api_key(), "anthropic")
    return Anthropic(api_key=api_key)


client = LazyClient(_create_anthropic_client, label="anthropic")

# Global verbose flag for retry logging
VERBOSE = False


def _parse_wait_time_from_error(error: Exception) -> float | None:
    """Parse wait time from Anthropic rate limit error message.

    Anthropic may provide retry timing in error messages or headers.
    This function attempts to extract it, but may return None to trigger
    exponential backoff.

    Args:
        error: The exception from Anthropic API

    Returns:
        Wait time in seconds if found, None otherwise
    """
    error_str = str(error)
    match = re.search(r"retry.*?(\d+(?:\.\d+)?)\s*(?:seconds?|s)", error_str.lower())
    if match:
        try:
            wait_time = float(match.group(1))
            return max(wait_time, 0.1)
        except ValueError:
            pass
    return None


def set_rate_limiter_verbose(verbose: bool) -> None:
    """Enable or disable verbose logging for Anthropic retry logic.

    Args:
        verbose: If True, print detailed logging about retry attempts and wait times
    """
    global VERBOSE
    VERBOSE = verbose


def ask_anthropic[T](
    user_msg: str,
    response_type: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = ClaudeModels.haiku45,
    *,
    client_override: "Anthropic | None" = None,
    record_store: "RecordStore | None" = None,
    temperature: float | None = None,
) -> T:
    """Call Anthropic API with structured output.

    Uses the structured outputs beta (constrained decoding, guaranteed valid JSON)
    when SDK >= 0.74.1. Retries on rate limit errors.

    If response_type is str or None, returns plain text.
    """
    from anthropic import APIError, RateLimitError

    max_attempts = 100
    api_client = client_override or client  # type: ignore[assignment]
    is_plain_text = response_type is str or response_type is None
    use_beta = _USE_STRUCTURED_OUTPUTS_BETA and not is_plain_text

    messages = [{"role": "user", "content": user_msg}]
    total_tpm_wait = 0.0
    started_at = datetime.now(UTC)

    for attempt in range(max_attempts):
        try:
            if VERBOSE and attempt > 0:
                print(
                    f"[Anthropic Retry] Attempt {attempt + 1}/{max_attempts} for model {model}"
                )

            if use_beta:
                # Structured outputs beta: guaranteed schema-valid JSON
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "betas": ["structured-outputs-2025-11-13"],
                    "output_format": response_type,
                    # max number allowed without streaming API
                    "max_tokens": 21_000,
                }
                if sys_msg is not None:
                    # Beta API requires system as list of content blocks
                    kwargs["system"] = [{"type": "text", "text": sys_msg}]
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = api_client.beta.messages.parse(**kwargs)
            else:
                # Plain text
                kwargs = {
                    "model": model,
                    "max_tokens": 4096,
                    "messages": messages,
                }
                if sys_msg is not None:
                    kwargs["system"] = sys_msg
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = api_client.messages.create(**kwargs)

            ended_at = datetime.now(UTC)
            usage = _extract_anthropic_usage(response, model=model)

            from covenance.record import record_llm_call

            record_llm_call(
                model=model,
                provider="anthropic",
                usage=usage,
                tpm_retry_wait_seconds=total_tpm_wait,
                started_at=started_at,
                ended_at=ended_at,
                record_store=record_store,
            )

            if VERBOSE and attempt > 0:
                print(
                    f"[Anthropic Retry] ✓ Successfully completed after {attempt + 1} attempt(s)"
                )

            if use_beta:
                # Beta returns parsed_output directly
                if response.parsed_output is None:
                    raise StructuredOutputParsingError(
                        f"Anthropic API returned None parsed_output. "
                        f"Model: {model}, response_type: {response_type}"
                    )
                return response.parsed_output

            # Plain text response
            if not response.content:
                raise StructuredOutputParsingError(
                    f"Anthropic API returned empty content. "
                    f"Model: {model}, response_type: {response_type}"
                )
            return response.content[0].text  # type: ignore[return-value]

        except RateLimitError as e:
            if attempt == max_attempts - 1:
                if VERBOSE:
                    print(f"[Anthropic Retry] ✗ Failed after {max_attempts} attempts")
                raise

            explicit_wait = _parse_wait_time_from_error(e)
            wait_time = explicit_wait if explicit_wait else exponential_backoff(attempt)

            if VERBOSE:
                print(
                    f"[Anthropic Retry] Rate limit (attempt {attempt + 1}/{max_attempts}): "
                    f"waiting {wait_time:.2f}s"
                )

            time.sleep(wait_time)
            total_tpm_wait += wait_time

        except APIError as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()

            if not is_rate_limit:
                if VERBOSE:
                    print(f"[Anthropic Retry] Non-rate-limit error: {type(e).__name__}")
                raise

            if attempt == max_attempts - 1:
                if VERBOSE:
                    print(f"[Anthropic Retry] ✗ Failed after {max_attempts} attempts")
                raise

            explicit_wait = _parse_wait_time_from_error(e)
            wait_time = explicit_wait if explicit_wait else exponential_backoff(attempt)

            if VERBOSE:
                print(
                    f"[Anthropic Retry] Rate limit (attempt {attempt + 1}/{max_attempts}): "
                    f"waiting {wait_time:.2f}s"
                )

            time.sleep(wait_time)
            total_tpm_wait += wait_time


def _extract_anthropic_usage(response, model: str) -> TokenUsage:
    """Extract token usage from Anthropic response and record to global stats."""
    if not hasattr(response, "usage") or response.usage is None:
        raise AttributeError(
            "Anthropic response missing usage information. Expected 'usage' attribute."
        )

    usage_obj = response.usage

    # Anthropic may omit cache fields if not used. We use isinstance to avoid MagicMock auto-creation in tests.
    cached_val = getattr(usage_obj, "cache_read_input_tokens", None)
    cached_tokens = cached_val if isinstance(cached_val, int) else 0

    usage = TokenUsage(
        prompt_tokens=usage_obj.input_tokens,
        completion_tokens=usage_obj.output_tokens,
        total_tokens=usage_obj.input_tokens + usage_obj.output_tokens,
        cached_tokens=cached_tokens,
    )

    return usage


if __name__ == "__main__":
    from pydantic import BaseModel

    class MovieReview(BaseModel):
        movie_title: str
        sentiment: str
        rating: float
        key_themes: list[str]

    result = ask_anthropic(
        user_msg="Review the movie 'Inception' by Christopher Nolan.",
        response_type=MovieReview,
        model=ClaudeModels.haiku45,
    )

    print(f"Movie: {result.movie_title}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Rating: {result.rating}/10")
    print(f"Themes: {', '.join(result.key_themes)}")
