"""Anthropic Claude client with structured output support and automatic retry."""

import random
import time
from datetime import UTC, datetime
from enum import Enum
from typing import TypeVar

from anthropic import Anthropic, APIError, RateLimitError
from pydantic import BaseModel

from ._lazy_client import LazyClient
from .exceptions import StructuredOutputParsingError
from .keys import get_anthropic_api_key, require_api_key
from .usage import TokenUsage

T = TypeVar("T")


def _create_anthropic_client() -> Anthropic:
    api_key = require_api_key(
        get_anthropic_api_key(), "anthropic", ["ANTHROPIC_API_KEY"]
    )
    return Anthropic(api_key=api_key)


client = LazyClient(_create_anthropic_client, label="anthropic")

# Global verbose flag for retry logging
VERBOSE = False


class ClaudeModels(str, Enum):
    """Anthropic Claude model identifiers.

    See: https://docs.anthropic.com/claude/docs/models-overview
    """

    # Latest models
    opus = "claude-opus-4-5"
    sonnet = "claude-sonnet-4-5"
    haiku = "claude-haiku-4-5"


def _pydantic_to_json_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to JSON schema for Anthropic tools.

    Args:
        model: Pydantic model class

    Returns:
        JSON schema dictionary compatible with Anthropic tools API
    """
    # Get the JSON schema from Pydantic
    schema = model.model_json_schema()
    # Anthropic expects the schema directly, not wrapped in a $defs structure
    # Remove $defs and inline references if present
    if "$defs" in schema:
        # For simplicity, we'll use the schema as-is and let Anthropic handle it
        # In practice, Anthropic should handle $ref references
        pass
    return schema


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
    jitter_factor = 0.5 + (random.random() * 0.5)
    wait_time = capped_wait * jitter_factor

    return max(wait_time, 0.1)  # Ensure minimum wait time


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
    # Look for common patterns in error messages
    # Anthropic may include retry-after information
    import re

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


def ask_anthropic_structured[T](
    user_msg: str,
    format: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = ClaudeModels.haiku,
) -> T:
    """Call Anthropic API with structured output using tools parameter.

    Uses Anthropic's tools parameter with JSON schema derived from Pydantic model
    to get structured output. Retries up to 100 times when encountering rate limit errors.

    If format is str, performs a standard chat completion and returns the text.

    Args:
        user_msg: User message/prompt
        format: Pydantic model class for structured output, or str/None for plain text
        sys_msg: Optional system message/instructions
        model: Claude model identifier (defaults to claude-3-5-haiku)

    Returns:
        Parsed Pydantic object of type T, or str if format is str

    Raises:
        StructuredOutputParsingError: If parsing fails
        Exception: After max_attempts retries on rate limit errors
    """
    max_attempts = 100

    # Handle plain text output
    is_plain_text = format is str or format is None

    if not is_plain_text:
        # Convert Pydantic model to JSON schema
        json_schema = _pydantic_to_json_schema(format)  # type: ignore[arg-type]

        # Create tool definition for structured output
        tool_name = (
            format.__name__ if hasattr(format, "__name__") else "structured_output"
        )
        tools = [
            {
                "name": tool_name,
                "description": f"Generate output matching the {tool_name} schema",
                "input_schema": json_schema,
            }
        ]
    else:
        tools = None
        tool_name = None

    # Build messages array
    messages = [{"role": "user", "content": user_msg}]

    total_tpm_wait = 0.0  # Accumulate TPM retry wait time
    started_at = datetime.now(UTC)  # Record absolute start time

    for attempt in range(max_attempts):
        try:
            if VERBOSE and attempt > 0:
                print(
                    f"[Anthropic Retry] Attempt {attempt + 1}/{max_attempts} for model {model}"
                )

            # Call Anthropic API
            api_kwargs = {
                "model": model,
                "max_tokens": 4096,
                "messages": messages,
            }
            if not is_plain_text:
                api_kwargs["tools"] = tools
                api_kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

            if sys_msg is not None:
                api_kwargs["system"] = sys_msg

            response = client.messages.create(**api_kwargs)

            ended_at = datetime.now(UTC)  # Record absolute end time
            usage = _extract_anthropic_usage(response, model=model)

            # Record to metrics context for DB persistence
            # Duration is calculated from started_at and ended_at timestamps
            from .metrics import record_llm_call

            record_llm_call(
                model=model,
                provider="anthropic",
                usage=usage,
                tpm_retry_wait_seconds=total_tpm_wait,
                started_at=started_at,
                ended_at=ended_at,
            )

            if VERBOSE and attempt > 0:
                print(
                    f"[Anthropic Retry] ✓ Successfully completed after {attempt + 1} attempt(s)"
                )

            if is_plain_text:
                if not response.content:
                    raise StructuredOutputParsingError(
                        f"Anthropic API returned empty content. "
                        f"Model: {model}, Format: {format}"
                    )
                return response.content[0].text  # type: ignore[return-value]

            # Extract structured output from tool use
            if not response.content:
                raise StructuredOutputParsingError(
                    f"Anthropic API returned empty content. "
                    f"Model: {model}, Format: {format}"
                )

            # Find the tool use block
            tool_use_block = None
            for block in response.content:
                if block.type == "tool_use" and block.name == tool_name:
                    tool_use_block = block
                    break

            if tool_use_block is None:
                raise StructuredOutputParsingError(
                    f"Anthropic API did not return tool_use block. "
                    f"Model: {model}, Format: {format}, Content: {response.content}"
                )

            # Parse the input as JSON and validate against Pydantic model
            try:
                parsed_data = tool_use_block.input
                # Validate and create Pydantic instance
                parsed = format(**parsed_data)
                return parsed
            except Exception as e:
                raise StructuredOutputParsingError(
                    f"Failed to parse Anthropic response as {format.__name__}: {e}. "
                    f"Model: {model}, Input: {tool_use_block.input}"
                ) from e

        except RateLimitError as e:
            if attempt == max_attempts - 1:
                if VERBOSE:
                    print(f"[Anthropic Retry] ✗ Failed after {max_attempts} attempts")
                raise

            # Try to parse wait time from error message first
            explicit_wait = _parse_wait_time_from_error(e)
            if explicit_wait is not None:
                wait_time = explicit_wait
                if VERBOSE:
                    print(
                        f"[Anthropic Retry] Rate limit error (attempt {attempt + 1}/{max_attempts}): "
                        f"using explicit wait time {wait_time:.2f}s from error message"
                    )
            else:
                # Use exponential backoff with jitter
                wait_time = _calculate_exponential_backoff(attempt)
                if VERBOSE:
                    print(
                        f"[Anthropic Retry] Rate limit error (attempt {attempt + 1}/{max_attempts}): "
                        f"exponential backoff wait {wait_time:.2f}s"
                    )

            if VERBOSE:
                error_str = str(e)
                if len(error_str) <= 300:
                    print(f"[Anthropic Retry] Error details: {error_str}")

            time.sleep(wait_time)
            total_tpm_wait += wait_time

        except APIError as e:
            # Handle other API errors
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()

            if not is_rate_limit:
                # Not a rate limit error, re-raise immediately
                if VERBOSE:
                    print(f"[Anthropic Retry] Non-rate-limit error: {type(e).__name__}")
                raise

            if attempt == max_attempts - 1:
                if VERBOSE:
                    print(f"[Anthropic Retry] ✗ Failed after {max_attempts} attempts")
                raise

            # Try to parse wait time from error message first
            explicit_wait = _parse_wait_time_from_error(e)
            if explicit_wait is not None:
                wait_time = explicit_wait
            else:
                wait_time = _calculate_exponential_backoff(attempt)

            if VERBOSE:
                print(
                    f"[Anthropic Retry] Rate limit error (attempt {attempt + 1}/{max_attempts}): "
                    f"waiting {wait_time:.2f}s before retry"
                )

            time.sleep(wait_time)
            total_tpm_wait += wait_time

        except Exception as e:
            # Handle other potential errors
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()

            if not is_rate_limit or attempt == max_attempts - 1:
                if VERBOSE:
                    print(
                        f"[Anthropic Retry] ✗ Unexpected error or max attempts reached: {type(e).__name__}"
                    )
                raise

            # Use exponential backoff
            wait_time = _calculate_exponential_backoff(attempt)

            if VERBOSE:
                print(
                    f"[Anthropic Retry] Unexpected rate limit error (attempt {attempt + 1}/{max_attempts}): "
                    f"waiting {wait_time:.2f}s before retry"
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

    # Record usage in global stats (for Python-side debugging)
    from .usage import usage_stats

    usage_stats.record_usage(usage, model=model, provider="anthropic")

    return usage


if __name__ == "__main__":
    from pydantic import BaseModel

    class MovieReview(BaseModel):
        movie_title: str
        sentiment: str
        rating: float
        key_themes: list[str]

    result = ask_anthropic_structured(
        user_msg="Review the movie 'Inception' by Christopher Nolan.",
        format=MovieReview,
        model=ClaudeModels.haiku,
    )

    print(f"Movie: {result.movie_title}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Rating: {result.rating}/10")
    print(f"Themes: {', '.join(result.key_themes)}")
