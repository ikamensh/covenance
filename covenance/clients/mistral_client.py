"""Mistral AI client with structured output support and automatic retry.

Note on structured output reliability:
Mistral uses probabilistic JSON generation, not constrained decoding like OpenAI.
This means structured output may occasionally produce invalid JSON and require retry.
Each retry attempt is recorded honestly as a separate LLM call.
"""

import json
import time
from typing import TYPE_CHECKING, TypeVar

from covenance._lazy_client import LazyClient
from covenance.exceptions import StructuredOutputParsingError, require_provider
from covenance.keys import get_mistral_api_key, require_api_key
from covenance.models import MistralModels
from covenance.record import TokenUsage
from covenance.retry import exponential_backoff

if TYPE_CHECKING:
    from mistralai import Mistral

    from covenance.record import RawCallResult

T = TypeVar("T")


def _create_mistral_client() -> "Mistral":
    require_provider("mistral")
    from mistralai import Mistral

    api_key = require_api_key(get_mistral_api_key(), "mistral")
    return Mistral(api_key=api_key)


client = LazyClient(_create_mistral_client, label="mistral")

# Global verbose flag for retry logging
VERBOSE = False


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


JSON_PARSE_MAX_RETRIES = 3  # Retries for Mistral's probabilistic JSON output


def ask_mistral[T](
    user_msg: str,
    response_type: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = MistralModels.small.value,
    *,
    client_override: "Mistral | None" = None,
    temperature: float | None = None,
) -> "RawCallResult":
    """Call Mistral API with structured output using native parse method. Returns RawCallResult.

    Uses Mistral's native client.chat.parse() method to get structured Pydantic
    output directly. Retries up to 100 times when encountering rate limit errors.

    Note: Mistral uses probabilistic JSON generation (not constrained decoding),
    so structured output may occasionally fail with JSONDecodeError. We retry
    up to JSON_PARSE_MAX_RETRIES times for such errors. Since the SDK doesn't
    provide token usage on parse failure, wasted tokens are estimated by
    multiplying successful attempt's tokens by the number of JSON retries.

    If response_type is str or None, performs a standard chat completion and returns the text.
    Raises StructuredOutputParsingError (with usage) on parse failure.
    """
    from mistralai.models import HTTPValidationError, SDKError

    from covenance.record import RawCallResult

    max_attempts = 100
    api_client = client_override or client  # type: ignore[assignment]

    messages = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": user_msg})

    total_tpm_wait = 0.0
    tpm_retries = 0
    json_parse_attempts = 0

    is_plain_text = response_type is str or response_type is None

    for attempt in range(max_attempts):
        try:
            if VERBOSE and attempt > 0:
                print(
                    f"[Mistral Retry] Attempt {attempt + 1}/{max_attempts} for model {model}"
                )

            if is_plain_text:
                response = api_client.chat.complete(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            else:
                response = api_client.chat.parse(
                    model=model,
                    messages=messages,
                    response_format=response_type,
                    temperature=temperature,
                )

            usage = _extract_mistral_usage(response, model=model)

            if VERBOSE and attempt > 0:
                print(
                    f"[Mistral Retry] ✓ Successfully completed after {attempt + 1} attempt(s)"
                )

            if is_plain_text:
                content = response.choices[0].message.content
                if content is None:
                    raise StructuredOutputParsingError(
                        f"Mistral API returned response but content field is None. "
                        f"Model: {model}, response_type: {response_type}",
                        usage=usage,
                    )
                output = content
            else:
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise StructuredOutputParsingError(
                        f"Mistral API returned response but parsed field is None. "
                        f"This may indicate a schema mismatch or parsing error. "
                        f"Model: {model}, response_type: {response_type}",
                        usage=usage,
                    )
                output = parsed

            return RawCallResult(
                output=output,
                usage=usage,
                tpm_retries=tpm_retries,
                tpm_wait_seconds=total_tpm_wait,
                json_retries=json_parse_attempts,
            )

        except StructuredOutputParsingError:
            raise  # Don't catch our own exceptions

        except json.JSONDecodeError as e:
            # Mistral's probabilistic JSON output occasionally produces invalid JSON
            json_parse_attempts += 1
            if json_parse_attempts >= JSON_PARSE_MAX_RETRIES:
                if VERBOSE:
                    print(
                        f"[Mistral Retry] JSON parse failed {json_parse_attempts} times, giving up"
                    )
                # No usage available on JSON parse failure
                raise StructuredOutputParsingError(
                    f"Mistral returned invalid JSON after {json_parse_attempts} attempts. "
                    f"Last error: {e}. Model: {model}, response_type: {response_type}",
                    usage=None,
                ) from e

            if VERBOSE:
                print(
                    f"[Mistral Retry] JSON parse error (attempt {json_parse_attempts}/{JSON_PARSE_MAX_RETRIES}): {e}"
                )
            time.sleep(0.5)
            continue

        except (SDKError, HTTPValidationError) as e:
            error_str = str(e)
            error_type = type(e).__name__

            is_rate_limit = False
            if isinstance(e, SDKError) and hasattr(e, "status_code"):
                is_rate_limit = e.status_code == 429
            if "429" in error_str or "rate limit" in error_str.lower():
                is_rate_limit = True

            if not is_rate_limit:
                if VERBOSE:
                    print(f"[Mistral Retry] Non-rate-limit error: {error_type}")
                raise

            if attempt == max_attempts - 1:
                if VERBOSE:
                    print(f"[Mistral Retry] ✗ Failed after {max_attempts} attempts")
                raise

            explicit_wait = _parse_wait_time_from_error(e)
            if explicit_wait is not None:
                wait_time = explicit_wait
                if VERBOSE:
                    print(
                        f"[Mistral Retry] Rate limit error (attempt {attempt + 1}/{max_attempts}): "
                        f"using explicit wait time {wait_time:.2f}s from error message"
                    )
            else:
                wait_time = exponential_backoff(attempt)
                if VERBOSE:
                    print(
                        f"[Mistral Retry] Rate limit error (attempt {attempt + 1}/{max_attempts}): "
                        f"exponential backoff wait {wait_time:.2f}s"
                    )

            if VERBOSE and len(error_str) <= 300:
                print(f"[Mistral Retry] Error details: {error_str}")

            time.sleep(wait_time)
            total_tpm_wait += wait_time
            tpm_retries += 1

        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()

            if not is_rate_limit or attempt == max_attempts - 1:
                if VERBOSE:
                    print(
                        f"[Mistral Retry] ✗ Unexpected error or max attempts reached: {type(e).__name__}"
                    )
                raise

            explicit_wait = _parse_wait_time_from_error(e)
            wait_time = (
                explicit_wait
                if explicit_wait is not None
                else exponential_backoff(attempt)
            )

            if VERBOSE:
                print(
                    f"[Mistral Retry] Unexpected rate limit error (attempt {attempt + 1}/{max_attempts}): "
                    f"waiting {wait_time:.2f}s before retry"
                )

            time.sleep(wait_time)
            total_tpm_wait += wait_time
            tpm_retries += 1

    raise RuntimeError("ask_mistral exhausted retry loop")


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

    return usage


if __name__ == "__main__":
    from pydantic import BaseModel

    class MovieReview(BaseModel):
        movie_title: str
        sentiment: str
        rating: float
        key_themes: list[str]

    raw = ask_mistral(
        user_msg="Review the movie 'Inception' by Christopher Nolan.",
        response_type=MovieReview,
        model=MistralModels.small.value,
    )
    result = raw.output

    print(f"Movie: {result.movie_title}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Rating: {result.rating}/10")
    print(f"Themes: {', '.join(result.key_themes)}")
