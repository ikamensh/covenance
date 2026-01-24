"""Mistral AI client with structured output support and automatic retry."""

import random
import time
from datetime import UTC, datetime
from enum import Enum
from typing import TypeVar

from mistralai import Mistral
from mistralai.models import HTTPValidationError, SDKError

from ._lazy_client import LazyClient
from .exceptions import StructuredOutputParsingError
from .keys import get_mistral_api_key, require_api_key
from .usage import TokenUsage

T = TypeVar("T")


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


def _create_mistral_client() -> Mistral:
    api_key = require_api_key(get_mistral_api_key(), "mistral", ["MISTRAL_API_KEY"])
    return Mistral(api_key=api_key)


client = LazyClient(_create_mistral_client, label="mistral")

# Global verbose flag for retry logging
VERBOSE = False


def _calculate_exponential_backoff(
    attempt: int, base_wait: float = 1.0, max_wait: float = 60.0
) -> float:
    """Calculate exponential backoff wait time with jitter.

    Uses exponential backoff: wait = base_wait * (2 ** attempt)
    Adds jitter to avoid thundering herd problem.
    Caps at max_wait to prevent excessive delays.

    Args:
        attempt: Current attempt number (0-indexed)
        base_wait: Base wait time in seconds (default: 1.0)
        max_wait: Maximum wait time in seconds (default: 60.0)

    Returns:
        Wait time in seconds with jitter applied
    """
    # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s, 60s, ...
    exponential_wait = base_wait * (2**attempt)

    # Cap at maximum wait time
    capped_wait = min(exponential_wait, max_wait)

    # Add jitter: randomize between 50% and 100% of calculated wait time
    # This helps avoid thundering herd when multiple requests retry simultaneously
    jitter_factor = 0.5 + (random.random() * 0.5)  # Random value between 0.5 and 1.0
    wait_time = capped_wait * jitter_factor

    return max(wait_time, 0.1)  # Ensure minimum wait time


def _parse_wait_time_from_error(error: Exception) -> float | None:
    """Parse wait time from Mistral rate limit error message.

    Note: As of 2025-11-23, Mistral does NOT provide retry timing in error messages.
    Observed error format:
        Status 429. Body: {
            "object":"error",
            "message":"Rate limit exceeded",
            "type":"rate_limited",
            "param":null,
            "code":"1300"
        }

    This function is kept as a placeholder in case Mistral adds Retry-After
    information in the future. Currently always returns None, triggering
    exponential backoff.

    Args:
        error: The exception from Mistral API

    Returns:
        None (Mistral doesn't provide explicit retry timing)
    """
    # Mistral does not provide Retry-After or explicit wait time in error messages
    # Tested 2025-11-23 with concurrent requests triggering 429 errors
    # Always use exponential backoff instead
    return None


def set_rate_limiter_verbose(verbose: bool) -> None:
    """Enable or disable verbose logging for Mistral retry logic.

    Args:
        verbose: If True, print detailed logging about retry attempts and wait times
    """
    global VERBOSE
    VERBOSE = verbose


def ask_mistral_structured[T](
    user_msg: str,
    format: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = MistralModels.small.value,
) -> T:
    """Call Mistral API with structured output using native parse method.

    Uses Mistral's native client.chat.parse() method to get structured Pydantic
    output directly. Retries up to 100 times when encountering rate limit errors.

    If format is str or None, performs a standard chat completion and returns the text.

    Args:
        user_msg: User message/prompt
        format: Pydantic model class for structured output, or str/None for plain text
        sys_msg: Optional system message/instructions
        model: Mistral model identifier (defaults to mistral-small-latest)

    Returns:
        Parsed Pydantic object of type T, or str if format is str

    Raises:
        Exception: After max_attempts retries on rate limit errors
    """
    max_attempts = 100

    # Build messages array
    messages = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": user_msg})

    total_tpm_wait = 0.0  # Accumulate TPM retry wait time
    started_at = datetime.now(UTC)  # Record absolute start time

    is_plain_text = format is str or format is None

    for attempt in range(max_attempts):
        try:
            if VERBOSE and attempt > 0:
                print(
                    f"[Mistral Retry] Attempt {attempt + 1}/{max_attempts} for model {model}"
                )

            if is_plain_text:
                # Use standard completion for plain text
                response = client.chat.complete(
                    model=model,
                    messages=messages,
                )
            else:
                # Use native structured output via chat.parse
                response = client.chat.parse(
                    model=model,
                    messages=messages,
                    response_format=format,
                    # temperature=0,
                )

            ended_at = datetime.now(UTC)  # Record absolute end time
            usage = _extract_mistral_usage(response, model=model)

            # Record to metrics context for DB persistence
            # Duration is calculated from started_at and ended_at timestamps
            from .metrics import record_llm_call

            record_llm_call(
                model=model,
                provider="mistral",
                usage=usage,
                tpm_retry_wait_seconds=total_tpm_wait,
                started_at=started_at,
                ended_at=ended_at,
            )

            if VERBOSE and attempt > 0:
                print(
                    f"[Mistral Retry] ✓ Successfully completed after {attempt + 1} attempt(s)"
                )

            if is_plain_text:
                content = response.choices[0].message.content
                if content is None:
                    raise StructuredOutputParsingError(
                        f"Mistral API returned response but content field is None. "
                        f"Model: {model}, Format: {format}"
                    )
                return content  # type: ignore[return-value]

            # Access the parsed Pydantic object
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise StructuredOutputParsingError(
                    f"Mistral API returned response but parsed field is None. "
                    f"This may indicate a schema mismatch or parsing error. "
                    f"Model: {model}, Format: {format}"
                )
            return parsed

        except (SDKError, HTTPValidationError) as e:
            # SDKError is the main rate limit error type from Mistral API
            # HTTPValidationError might also occur for validation issues
            error_str = str(e)
            error_type = type(e).__name__

            # Check if it's a rate limit error (status 429)
            is_rate_limit = False

            # Check for SDKError with status_code attribute
            if isinstance(e, SDKError) and hasattr(e, "status_code"):
                is_rate_limit = e.status_code == 429
            # Also check error message content
            if "429" in error_str or "rate limit" in error_str.lower():
                is_rate_limit = True

            if not is_rate_limit:
                # Not a rate limit error, re-raise immediately
                if VERBOSE:
                    print(f"[Mistral Retry] Non-rate-limit error: {error_type}")
                raise

            if attempt == max_attempts - 1:
                # Last attempt failed, re-raise the exception
                if VERBOSE:
                    print(f"[Mistral Retry] ✗ Failed after {max_attempts} attempts")
                raise

            # Try to parse wait time from error message first
            # If not found, use exponential backoff
            explicit_wait = _parse_wait_time_from_error(e)
            if explicit_wait is not None:
                wait_time = explicit_wait
                if VERBOSE:
                    print(
                        f"[Mistral Retry] Rate limit error (attempt {attempt + 1}/{max_attempts}): "
                        f"using explicit wait time {wait_time:.2f}s from error message"
                    )
            else:
                # Use exponential backoff with jitter
                wait_time = _calculate_exponential_backoff(attempt)
                if VERBOSE:
                    print(
                        f"[Mistral Retry] Rate limit error (attempt {attempt + 1}/{max_attempts}): "
                        f"exponential backoff wait {wait_time:.2f}s"
                    )

            if VERBOSE and len(error_str) <= 300:
                print(f"[Mistral Retry] Error details: {error_str}")

            time.sleep(wait_time)
            total_tpm_wait += wait_time

        except Exception as e:
            # Handle other potential errors (network, API changes, etc.)
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()

            if not is_rate_limit or attempt == max_attempts - 1:
                if VERBOSE:
                    print(
                        f"[Mistral Retry] ✗ Unexpected error or max attempts reached: {type(e).__name__}"
                    )
                raise

            # Try to parse wait time from error message first
            explicit_wait = _parse_wait_time_from_error(e)
            if explicit_wait is not None:
                wait_time = explicit_wait
            else:
                # Use exponential backoff
                wait_time = _calculate_exponential_backoff(attempt)

            if VERBOSE:
                print(
                    f"[Mistral Retry] Unexpected rate limit error (attempt {attempt + 1}/{max_attempts}): "
                    f"waiting {wait_time:.2f}s before retry"
                )

            time.sleep(wait_time)
            total_tpm_wait += wait_time


def _extract_mistral_usage(response, model: str) -> TokenUsage:
    """Extract token usage from Mistral response and record to global stats."""
    if not hasattr(response, "usage") or response.usage is None:
        raise AttributeError(
            "Mistral response missing usage information. Expected 'usage' attribute."
        )

    usage_obj = response.usage
    # DEBUG: print(f"DEBUG Mistral Usage for {model}: {usage_obj}")

    # Mistral does not support prompt caching, so cached_tokens stays 0
    usage = TokenUsage(
        prompt_tokens=usage_obj.prompt_tokens,
        completion_tokens=usage_obj.completion_tokens,
        total_tokens=usage_obj.total_tokens,
        cached_tokens=0,
    )

    # Record usage in global stats (for Python-side debugging)
    from .usage import usage_stats

    usage_stats.record_usage(usage, model=model, provider="mistral")

    return usage


if __name__ == "__main__":
    from pydantic import BaseModel

    class MovieReview(BaseModel):
        movie_title: str
        sentiment: str
        rating: float
        key_themes: list[str]

    result = ask_mistral_structured(
        user_msg="Review the movie 'Inception' by Christopher Nolan.",
        format=MovieReview,
        model=MistralModels.small.value,
    )

    print(f"Movie: {result.movie_title}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Rating: {result.rating}/10")
    print(f"Themes: {', '.join(result.key_themes)}")
