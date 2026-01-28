"""Unit tests for Mistral client implementation with mocked dependencies."""

from unittest.mock import MagicMock, patch

import pytest
from mistralai.models import HTTPValidationError, SDKError
from pydantic import BaseModel

from covenance.clients.mistral_client import (
    _extract_mistral_usage,
    _parse_wait_time_from_error,
    ask_mistral,
    set_rate_limiter_verbose,
)
from covenance.exceptions import StructuredOutputParsingError


class SampleResponse(BaseModel):
    """Sample response model for testing."""

    answer: str
    value: int


def test_parse_wait_time_from_error_always_returns_none():
    """Property test: Mistral error parser always returns None (no retry info in errors)."""
    # Create a mock error - SDKError doesn't take status_code in constructor
    error = Exception("Rate limit exceeded")
    wait_time = _parse_wait_time_from_error(error)

    assert wait_time is None


def test_set_rate_limiter_verbose():
    """Property test: sets verbose flag."""
    set_rate_limiter_verbose(True)
    # Can't easily test the global flag without side effects, but we can verify it doesn't crash
    set_rate_limiter_verbose(False)


@patch("covenance.clients.mistral_client.client")
@patch("covenance.record.record_llm_call")
@patch("covenance.clients.mistral_client.time.sleep")
def test_ask_mistral_plain_text(mock_sleep, mock_record, mock_client):
    """Property test: plain text requests return string."""
    # Mock response
    mock_message = MagicMock()
    mock_message.content = "Hello, world!"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    mock_client.chat.complete.return_value = mock_response

    result = ask_mistral("Hello", response_type=str)

    assert result == "Hello, world!"
    mock_client.chat.complete.assert_called_once()
    mock_record.assert_called_once()


@patch("covenance.clients.mistral_client.client")
@patch("covenance.record.record_llm_call")
@patch("covenance.clients.mistral_client.time.sleep")
def test_ask_mistral_structured_output(mock_sleep, mock_record, mock_client):
    """Property test: structured output requests return Pydantic model."""
    # Mock parsed response
    parsed_obj = SampleResponse(answer="test", value=42)

    mock_message = MagicMock()
    mock_message.parsed = parsed_obj

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    mock_client.chat.parse.return_value = mock_response

    result = ask_mistral("Hello", response_type=SampleResponse)

    assert isinstance(result, SampleResponse)
    assert result.answer == "test"
    assert result.value == 42
    mock_client.chat.parse.assert_called_once()


@patch("covenance.clients.mistral_client.client")
@patch("covenance.record.record_llm_call")
@patch("covenance.clients.mistral_client.time.sleep")
def test_ask_mistral_with_system_message(mock_sleep, mock_record, mock_client):
    """Property test: system message is included in messages array."""
    mock_message = MagicMock()
    mock_message.content = "Response"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    mock_client.chat.complete.return_value = mock_response

    ask_mistral("Hello", sys_msg="You are a helpful assistant", response_type=str)

    call_kwargs = mock_client.chat.complete.call_args[1]
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


@patch("covenance.clients.mistral_client.client")
@patch("covenance.record.record_llm_call")
@patch("covenance.clients.mistral_client.time.sleep")
def test_ask_mistral_retries_on_rate_limit(mock_sleep, mock_record, mock_client):
    """Property test: retries on rate limit errors."""
    # First call raises rate limit, second succeeds
    mock_message = MagicMock()
    mock_message.content = "Success"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    # Create SDKError with status_code attribute - SDKError needs raw_response
    # Use a real exception that can be raised
    class MockSDKError(SDKError):
        def __init__(self):
            super().__init__(message="Rate limit exceeded", raw_response=MagicMock())
            self.status_code = 429

    rate_limit_error = MockSDKError()
    mock_client.chat.complete.side_effect = [rate_limit_error, mock_response]

    result = ask_mistral("Hello", response_type=str)

    assert result == "Success"
    assert mock_client.chat.complete.call_count == 2
    assert mock_sleep.call_count == 1


@patch("covenance.clients.mistral_client.client")
@patch("covenance.record.record_llm_call")
@patch("covenance.clients.mistral_client.time.sleep")
def test_ask_mistral_raises_after_max_retries(mock_sleep, mock_record, mock_client):
    """Property test: raises exception after max retries."""

    class MockSDKError(SDKError):
        def __init__(self):
            super().__init__(message="Rate limit exceeded", raw_response=MagicMock())
            self.status_code = 429

    rate_limit_error = MockSDKError()
    mock_client.chat.complete.side_effect = rate_limit_error

    with pytest.raises(SDKError):
        ask_mistral("Hello", response_type=str)

    # Should attempt 100 times
    assert mock_client.chat.complete.call_count == 100


@patch("covenance.clients.mistral_client.client")
@patch("covenance.record.record_llm_call")
@patch("covenance.clients.mistral_client.time.sleep")
def test_ask_mistral_empty_content_raises(mock_sleep, mock_record, mock_client):
    """Property test: None content raises StructuredOutputParsingError."""
    mock_message = MagicMock()
    mock_message.content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock()

    mock_client.chat.complete.return_value = mock_response

    with pytest.raises(StructuredOutputParsingError, match="content field is None"):
        ask_mistral("Hello", response_type=str)


@patch("covenance.clients.mistral_client.client")
@patch("covenance.record.record_llm_call")
@patch("covenance.clients.mistral_client.time.sleep")
def test_ask_mistral_missing_parsed_raises(mock_sleep, mock_record, mock_client):
    """Property test: missing parsed field raises StructuredOutputParsingError."""
    mock_message = MagicMock()
    mock_message.parsed = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    mock_client.chat.parse.return_value = mock_response

    with pytest.raises(StructuredOutputParsingError, match="parsed field is None"):
        ask_mistral("Hello", response_type=SampleResponse)


@patch("covenance.clients.mistral_client.client")
@patch("covenance.record.record_llm_call")
@patch("covenance.clients.mistral_client.time.sleep")
def test_ask_mistral_non_rate_limit_error_raises_immediately(
    mock_sleep, mock_record, mock_client
):
    """Property test: non-rate-limit errors raise immediately."""
    # Create a mock HTTPValidationError
    validation_error = MagicMock(spec=HTTPValidationError)
    validation_error.__str__ = lambda self: "Invalid request"
    mock_client.chat.complete.side_effect = validation_error

    with pytest.raises(Exception):  # Will catch the MagicMock exception
        ask_mistral("Hello", response_type=str)

    # Should not retry
    assert mock_client.chat.complete.call_count == 1
    assert mock_sleep.call_count == 0


@patch("covenance.clients.mistral_client.client")
@patch("covenance.record.record_llm_call")
@patch("covenance.clients.mistral_client.time.sleep")
def test_ask_mistral_rate_limit_by_status_code(mock_sleep, mock_record, mock_client):
    """Property test: detects rate limit by status_code attribute."""
    mock_message = MagicMock()
    mock_message.content = "Success"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    class MockSDKError(SDKError):
        def __init__(self):
            super().__init__(message="Error", raw_response=MagicMock())
            self.status_code = 429

    rate_limit_error = MockSDKError()
    mock_client.chat.complete.side_effect = [rate_limit_error, mock_response]

    result = ask_mistral("Hello", response_type=str)

    assert result == "Success"
    assert mock_sleep.call_count == 1


def test_extract_mistral_usage():
    """Property test: extracts token usage from Mistral response."""
    mock_response = MagicMock()
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150

    usage = _extract_mistral_usage(mock_response, model="mistral-small-latest")

    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150
    assert usage.cached_tokens == 0  # Mistral doesn't support caching


def test_extract_mistral_usage_missing_usage_raises():
    """Property test: missing usage attribute raises AttributeError."""
    mock_response = MagicMock()
    del mock_response.usage

    with pytest.raises(AttributeError, match="missing usage information"):
        _extract_mistral_usage(mock_response, model="mistral-small-latest")


@patch("covenance.clients.mistral_client.client")
@patch("covenance.record.record_llm_call")
@patch("covenance.clients.mistral_client.time.sleep")
def test_ask_mistral_with_client_override(mock_sleep, mock_record, mock_client):
    """Property test: client_override parameter uses provided client."""
    override_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Response"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    override_client.chat.complete.return_value = mock_response

    result = ask_mistral("Hello", response_type=str, client_override=override_client)

    assert result == "Response"
    override_client.chat.complete.assert_called_once()
    mock_client.chat.complete.assert_not_called()
