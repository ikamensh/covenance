"""Unit tests for Anthropic client implementation with mocked dependencies.

Tests cover both the structured outputs beta path (SDK >= 0.74.1) and plain text.
"""

from unittest.mock import MagicMock, patch

import pytest
from anthropic import APIError, RateLimitError
from pydantic import BaseModel

from covenance.clients.anthropic_client import (
    _USE_STRUCTURED_OUTPUTS_BETA,
    _extract_anthropic_usage,
    _is_rate_limit_error,
    _parse_wait_time_from_error,
    ask_anthropic,
    set_rate_limiter_verbose,
)
from covenance.exceptions import StructuredOutputParsingError


class SampleResponse(BaseModel):
    """Sample response model for testing."""

    answer: str
    value: int


def test_is_rate_limit_error_detects_rate_limit():
    """Property: detects rate limit from RateLimitError or string indicators."""
    rate_limit_err = RateLimitError(
        message="Rate limit", response=MagicMock(status_code=429), body={}
    )
    assert _is_rate_limit_error(rate_limit_err) is True
    assert _is_rate_limit_error(Exception("Error 429 too many")) is True
    assert _is_rate_limit_error(Exception("rate limit exceeded")) is True
    assert _is_rate_limit_error(Exception("Some other error")) is False


def test_parse_wait_time_from_error_with_retry_info():
    """Property test: extracts wait time from error message when present."""
    error = Exception("Rate limit exceeded. Please retry after 5 seconds")
    wait_time = _parse_wait_time_from_error(error)

    assert wait_time == 5.0


def test_parse_wait_time_from_error_without_retry_info():
    """Property test: returns None when no retry info in error."""
    error = Exception("Rate limit exceeded")
    wait_time = _parse_wait_time_from_error(error)

    assert wait_time is None


def test_parse_wait_time_from_error_with_decimal():
    """Property test: handles decimal wait times."""
    error = Exception("Retry after 2.5 seconds")
    wait_time = _parse_wait_time_from_error(error)

    assert wait_time == 2.5


def test_parse_wait_time_from_error_minimum_wait():
    """Property test: enforces minimum wait time of 0.1 seconds."""
    error = Exception("Retry after 0.05 seconds")
    wait_time = _parse_wait_time_from_error(error)

    assert wait_time == 0.1


def test_set_rate_limiter_verbose():
    """Property test: sets verbose flag."""
    set_rate_limiter_verbose(True)
    # Can't easily test the global flag without side effects, but we can verify it doesn't crash
    set_rate_limiter_verbose(False)


@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_plain_text(mock_sleep, mock_record, mock_client):
    """Property test: plain text requests return string."""
    # Mock response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello, world!")]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.cache_read_input_tokens = None

    mock_client.messages.create.return_value = mock_response

    result = ask_anthropic("Hello", response_type=str)

    assert result.output == "Hello, world!"
    mock_client.messages.create.assert_called_once()


@pytest.mark.skipif(
    not _USE_STRUCTURED_OUTPUTS_BETA, reason="Requires SDK >= 0.74.1 for beta API"
)
@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_structured_output(mock_sleep, mock_record, mock_client):
    """Property test: structured output requests return Pydantic model via beta API."""
    # Mock response from beta.messages.parse
    mock_response = MagicMock()
    mock_response.parsed_output = SampleResponse(answer="test", value=42)
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.cache_read_input_tokens = None

    mock_client.beta.messages.parse.return_value = mock_response

    result = ask_anthropic("Hello", response_type=SampleResponse)

    assert isinstance(result.output, SampleResponse)
    assert result.output.answer == "test"
    assert result.output.value == 42
    mock_client.beta.messages.parse.assert_called_once()


@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_with_system_message(mock_sleep, mock_record, mock_client):
    """Property test: system message is included in API call."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Response")]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.cache_read_input_tokens = None

    mock_client.messages.create.return_value = mock_response

    ask_anthropic("Hello", sys_msg="You are a helpful assistant", response_type=str)

    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["system"] == "You are a helpful assistant"


@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_retries_on_rate_limit(mock_sleep, mock_record, mock_client):
    """Property test: retries on rate limit errors."""
    # First call raises rate limit, second succeeds
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Success")]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.cache_read_input_tokens = None

    # Create proper RateLimitError - it needs response and body kwargs
    rate_limit_error = RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body={},
    )
    mock_client.messages.create.side_effect = [rate_limit_error, mock_response]

    result = ask_anthropic("Hello", response_type=str)

    assert result.output == "Success"
    assert mock_client.messages.create.call_count == 2
    assert mock_sleep.call_count == 1


@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_raises_after_max_retries(mock_sleep, mock_record, mock_client):
    """Property test: raises exception after max retries."""
    rate_limit_error = RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body={},
    )
    mock_client.messages.create.side_effect = rate_limit_error

    with pytest.raises(RateLimitError):
        ask_anthropic("Hello", response_type=str)

    # Should attempt 100 times
    assert mock_client.messages.create.call_count == 100


@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_empty_content_raises(mock_sleep, mock_record, mock_client):
    """Property test: empty content raises StructuredOutputParsingError."""
    mock_response = MagicMock()
    mock_response.content = []
    mock_response.usage = MagicMock()

    mock_client.messages.create.return_value = mock_response

    with pytest.raises(StructuredOutputParsingError, match="empty content"):
        ask_anthropic("Hello", response_type=str)


@pytest.mark.skipif(
    not _USE_STRUCTURED_OUTPUTS_BETA, reason="Requires SDK >= 0.74.1 for beta API"
)
@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_none_parsed_output_raises(mock_sleep, mock_record, mock_client):
    """Property test: None parsed_output raises StructuredOutputParsingError."""
    mock_response = MagicMock()
    mock_response.parsed_output = None
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.cache_read_input_tokens = None

    mock_client.beta.messages.parse.return_value = mock_response

    with pytest.raises(StructuredOutputParsingError, match="None parsed_output"):
        ask_anthropic("Hello", response_type=SampleResponse)


@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_non_rate_limit_error_raises_immediately(
    mock_sleep, mock_record, mock_client
):
    """Property test: non-rate-limit errors raise immediately."""
    # Create a mock APIError - APIError needs specific constructor args
    # Use MagicMock to simulate the error
    api_error = MagicMock(spec=APIError)
    api_error.__str__ = lambda self: "Invalid API key"
    mock_client.messages.create.side_effect = api_error

    with pytest.raises(Exception):  # Will catch the MagicMock exception
        ask_anthropic("Hello", response_type=str)

    # Should not retry (non-rate-limit errors raise immediately)
    assert mock_client.messages.create.call_count == 1
    assert mock_sleep.call_count == 0


def test_extract_anthropic_usage():
    """Property test: extracts token usage from Anthropic response."""
    mock_response = MagicMock()
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_response.usage.cache_read_input_tokens = 10

    usage = _extract_anthropic_usage(mock_response, model="claude-3-5-haiku")

    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150
    assert usage.cached_tokens == 10


def test_extract_anthropic_usage_without_cache():
    """Property test: handles missing cache tokens."""
    mock_response = MagicMock()
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    # Simulate missing cache_read_input_tokens
    del mock_response.usage.cache_read_input_tokens

    usage = _extract_anthropic_usage(mock_response, model="claude-3-5-haiku")

    assert usage.cached_tokens == 0


def test_extract_anthropic_usage_missing_usage_raises():
    """Property test: missing usage attribute raises AttributeError."""
    mock_response = MagicMock()
    del mock_response.usage

    with pytest.raises(AttributeError, match="missing usage information"):
        _extract_anthropic_usage(mock_response, model="claude-3-5-haiku")


@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_with_client_override(mock_sleep, mock_record, mock_client):
    """Property test: client_override parameter uses provided client."""
    override_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Response")]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.cache_read_input_tokens = None
    override_client.messages.create.return_value = mock_response

    result = ask_anthropic("Hello", response_type=str, client_override=override_client)

    assert result.output == "Response"
    override_client.messages.create.assert_called_once()
    mock_client.messages.create.assert_not_called()


@pytest.mark.skipif(
    _USE_STRUCTURED_OUTPUTS_BETA, reason="Only test tool-use fallback when beta is not available"
)
@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_tool_use_fallback(mock_sleep, mock_record, mock_client):
    """Property test: structured output uses tool-use when beta is not available."""
    # Mock response with tool_use block
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "SampleResponse"
    tool_use_block.input = {"answer": "test", "value": 42}

    mock_response = MagicMock()
    mock_response.content = [tool_use_block]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.cache_read_input_tokens = None

    mock_client.messages.create.return_value = mock_response

    result = ask_anthropic("Hello", response_type=SampleResponse)

    assert isinstance(result, SampleResponse)
    assert result.answer == "test"
    assert result.value == 42

    # Verify tool-use was called
    call_kwargs = mock_client.messages.create.call_args[1]
    assert "tools" in call_kwargs
    assert "tool_choice" in call_kwargs
    assert call_kwargs["tool_choice"]["name"] == "SampleResponse"
    mock_record.assert_called_once()


@pytest.mark.skipif(
    _USE_STRUCTURED_OUTPUTS_BETA, reason="Only test tool-use fallback when beta is not available"
)
@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_tool_use_missing_block_raises(mock_sleep, mock_record, mock_client):
    """Property test: missing tool_use block raises StructuredOutputParsingError."""
    # Mock response without tool_use block
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Some text")]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.cache_read_input_tokens = None

    mock_client.messages.create.return_value = mock_response

    with pytest.raises(StructuredOutputParsingError, match="No tool_use block"):
        ask_anthropic("Hello", response_type=SampleResponse)


@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_retries_on_unexpected_rate_limit_error(
    mock_sleep, mock_record, mock_client
):
    """Property test: best-effort retry on unexpected Exception with rate limit indicators."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Success")]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.cache_read_input_tokens = None

    # Simulate unexpected exception with rate limit indicator in message
    unexpected_error = Exception("Connection failed: 429 too many requests")
    mock_client.messages.create.side_effect = [unexpected_error, mock_response]

    result = ask_anthropic("Hello", response_type=str)

    assert result.output == "Success"
    assert mock_client.messages.create.call_count == 2
    assert mock_sleep.call_count == 1


@patch("covenance.clients.anthropic_client.client")
@patch("covenance.record.record_llm_call", autospec=True)
@patch("covenance.clients.anthropic_client.time.sleep", autospec=True)
def test_ask_anthropic_unexpected_non_rate_limit_error_raises_immediately(
    mock_sleep, mock_record, mock_client
):
    """Property test: unexpected non-rate-limit errors raise immediately."""
    unexpected_error = Exception("Some random network error")
    mock_client.messages.create.side_effect = unexpected_error

    with pytest.raises(Exception, match="network error"):
        ask_anthropic("Hello", response_type=str)

    # Should not retry
    assert mock_client.messages.create.call_count == 1
    assert mock_sleep.call_count == 0
