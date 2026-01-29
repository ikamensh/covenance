import re
import time
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypeVar

from covenance._lazy_client import LazyClient
from covenance.exceptions import StructuredOutputParsingError, require_provider
from covenance.keys import get_gemini_api_key, require_api_key
from covenance.models import GeminiModels
from covenance.record import TokenUsage

if TYPE_CHECKING:
    from google import genai

    from covenance.record import RecordStore

# Suppress warning about non-text parts (thought_signature) in Gemini responses.
# This is expected when using structured outputs - the library handles it automatically
# via response.parsed, so the warning is just informational.
warnings.filterwarnings(
    "ignore",
    message=".*non-text parts.*",
)

T = TypeVar("T")


def _create_gemini_client() -> "genai.Client":
    require_provider("google")
    from google import genai

    api_key = require_api_key(get_gemini_api_key(), "gemini")
    return genai.Client(api_key=api_key)


client = LazyClient(_create_gemini_client, label="gemini")

# Global verbose flag for retry logging
VERBOSE = False


def _parse_wait_time_from_error(error: Exception) -> float:
    """Parse wait time from Gemini ClientError message.

    The error message typically contains: "Please retry in X.XXXs."
    Also checks for retryDelay in error details.

    Args:
        error: The ClientError exception

    Returns:
        Wait time in seconds, or 1.0 if parsing fails
    """
    error_message = str(error)

    # First, try to parse from "Please retry in X.XXXs." pattern
    match = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", error_message)
    if match:
        try:
            wait_time = float(match.group(1))
            return max(wait_time, 0.1)
        except ValueError:
            pass

    # Fallback: try to extract from error details if available
    if hasattr(error, "error") and isinstance(error.error, dict):
        details = error.error.get("details", [])
        for detail in details:
            if (
                isinstance(detail, dict)
                and detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo"
            ):
                retry_delay = detail.get("retryDelay", "")
                # Parse duration like "51s" or "51.225s"
                match = re.search(r"([0-9]+(?:\.[0-9]+)?)s", retry_delay)
                if match:
                    try:
                        wait_time = float(match.group(1))
                        return max(wait_time, 0.1)
                    except ValueError:
                        pass

    # Fallback: use a default wait time
    return 1.0


def set_rate_limiter_verbose(verbose: bool) -> None:
    """Enable or disable verbose logging for Google/Gemini retry logic.

    Args:
        verbose: If True, print detailed logging about retry attempts and wait times
    """
    global VERBOSE
    VERBOSE = verbose


def ask_gemini[T](
    user_msg: str,
    response_type: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = GeminiModels.flash_25.value,
    *,
    client_override: "genai.Client | None" = None,
    record_store: "RecordStore | None" = None,
    temperature: float | None = None,
) -> T:
    """Call Gemini API with automatic retry on rate limit errors.

    Retries up to 100 times when encountering 429 RESOURCE_EXHAUSTED errors,
    parsing the wait time from the error message and waiting accordingly.

    If response_type is str or None, performs a standard chat completion and returns the text.
    """
    from google.genai.errors import ClientError

    max_attempts = 100
    api_client = client_override or client  # type: ignore[assignment]
    total_tpm_wait = 0.0  # Accumulate TPM retry wait time
    started_at = datetime.now(UTC)  # Record absolute start time

    is_plain_text = response_type is str or response_type is None

    cfg: dict = {}
    if is_plain_text:
        cfg["response_mime_type"] = "text/plain"
    else:
        cfg["response_mime_type"] = "application/json"
        cfg["response_schema"] = response_type  # lets Gemini auto-validate & parse

    if sys_msg:
        cfg["system_instruction"] = sys_msg

    if temperature is not None:
        cfg["temperature"] = temperature

    for attempt in range(max_attempts):
        try:
            if VERBOSE and attempt > 0:
                print(
                    f"[Gemini Retry] Attempt {attempt + 1}/{max_attempts} for model {model}"
                )

            response = api_client.models.generate_content(
                model=model,
                contents=user_msg,  # multi-turn = list[dict]; single string is fine here
                config=cfg,
            )
            ended_at = datetime.now(UTC)  # Record absolute end time
            usage = _extract_gemini_usage(response, model=model)

            from covenance.record import record_llm_call

            record_llm_call(
                model=model,
                provider="gemini",
                usage=usage,
                tpm_retry_wait_seconds=total_tpm_wait,
                started_at=started_at,
                ended_at=ended_at,
                record_store=record_store,
            )

            if VERBOSE and attempt > 0:
                print(
                    f"[Gemini Retry] Successfully completed after {attempt + 1} attempt(s)"
                )

            if is_plain_text:
                if response.text is None:
                    raise StructuredOutputParsingError(
                        f"Gemini API returned response but text field is None. "
                        f"Model: {model}, response_type: {response_type}"
                    )
                return response.text  # type: ignore[return-value]

            if response.parsed is None:
                raise StructuredOutputParsingError(
                    f"Gemini API returned response but parsed field is None. "
                    f"This may indicate a schema mismatch or parsing error. "
                    f"Model: {model}, response_type: {response_type}"
                )

            return response.parsed
        except ClientError as e:
            # Check if it's a 429 RESOURCE_EXHAUSTED error
            # ClientError can store error info in different ways
            is_rate_limit = False
            error_str = str(e)

            # Check status_code attribute (if it exists)
            if hasattr(e, "status_code") and e.status_code == 429:
                is_rate_limit = True
            # Check error dict attribute (if it exists)
            if hasattr(e, "error") and isinstance(e.error, dict):
                error_status = e.error.get("status", "")
                error_code = e.error.get("code", 0)
                if error_status == "RESOURCE_EXHAUSTED" or error_code == 429:
                    is_rate_limit = True
            # Check error message string - most reliable fallback
            # The error message contains "429 RESOURCE_EXHAUSTED" and optionally "Please retry in X.XXXs"
            if "RESOURCE_EXHAUSTED" in error_str and "429" in error_str:
                is_rate_limit = True

            if not is_rate_limit:
                # Not a rate limit error, re-raise immediately
                if VERBOSE:
                    print(
                        f"[Gemini Retry] Non-rate-limit ClientError: {error_str[:200]}"
                    )
                raise

            if attempt == max_attempts - 1:
                # Last attempt failed, re-raise the exception
                if VERBOSE:
                    print(f"[Gemini Retry] Failed after {max_attempts} attempts")
                raise

            wait_time = _parse_wait_time_from_error(e)
            # Add a small buffer to the wait time
            wait_time = max(wait_time, 1.0)

            if VERBOSE:
                error_msg = str(e)
                print(
                    f"[Gemini Retry] Rate limit error (attempt {attempt + 1}/{max_attempts}): "
                    f"waiting {wait_time:.2f}s before retry"
                )
                if len(error_msg) <= 300:
                    print(f"[Gemini Retry] Error details: {error_msg}")

            time.sleep(wait_time)
            total_tpm_wait += wait_time
            # Continue to next attempt


def _extract_gemini_usage(response, model: str) -> TokenUsage:
    """Extract token usage from Gemini response and record to global stats."""
    usage_metadata = response.usage_metadata

    usage = TokenUsage(
        prompt_tokens=usage_metadata.prompt_token_count,
        completion_tokens=usage_metadata.candidates_token_count,
        total_tokens=usage_metadata.total_token_count,
        cached_tokens=usage_metadata.cached_content_token_count or 0,
    )

    return usage
