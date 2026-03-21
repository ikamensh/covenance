"""Mock tests for mistral_client.py covering structured output path and uncovered branches."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from covenance.clients.mistral_client import (
    _extract_mistral_usage,
    _parse_wait_time_from_error,
    ask_mistral,
    set_rate_limiter_verbose,
)
from covenance.exceptions import StructuredOutputParsingError


class SampleResponse(BaseModel):
    answer: str
    value: int


def _mock_mistral_response(content=None, parsed=None):
    """Create a mock Mistral response."""
    msg = MagicMock()
    msg.content = content
    msg.parsed = parsed
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    return resp


# --- structured output (chat.parse) ---


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_structured_output_via_parse(mock_sleep, mock_client):
    """Structured output uses chat.parse with response_format."""
    parsed = SampleResponse(answer="Paris", value=95)
    resp = _mock_mistral_response(parsed=parsed)
    mock_client.chat.parse.return_value = resp

    result = ask_mistral("Q", response_type=SampleResponse)

    assert result.output.answer == "Paris"
    mock_client.chat.parse.assert_called_once()
    call_kwargs = mock_client.chat.parse.call_args[1]
    assert call_kwargs["response_format"] is SampleResponse


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_structured_output_none_parsed_raises(mock_sleep, mock_client):
    """None parsed field raises StructuredOutputParsingError."""
    resp = _mock_mistral_response(parsed=None)
    mock_client.chat.parse.return_value = resp

    with pytest.raises(StructuredOutputParsingError, match="parsed field is None"):
        ask_mistral("Q", response_type=SampleResponse)


# --- sys_msg ---


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_sys_msg_prepended(mock_sleep, mock_client):
    """System message appears first in messages list."""
    resp = _mock_mistral_response(content="OK")
    mock_client.chat.complete.return_value = resp

    ask_mistral("Hello", response_type=str, sys_msg="Be concise")

    call_kwargs = mock_client.chat.complete.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "Be concise"}
    assert messages[1] == {"role": "user", "content": "Hello"}


# --- client_override ---


@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_client_override(mock_sleep):
    override = MagicMock()
    resp = _mock_mistral_response(content="From override")
    override.chat.complete.return_value = resp

    result = ask_mistral("Hello", response_type=str, client_override=override)

    assert result.output == "From override"
    override.chat.complete.assert_called_once()


# --- None content raises ---


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_none_content_raises(mock_sleep, mock_client):
    resp = _mock_mistral_response(content=None)
    mock_client.chat.complete.return_value = resp

    with pytest.raises(StructuredOutputParsingError, match="content field is None"):
        ask_mistral("Hello", response_type=str)


# --- SDKError with status_code=429 ---


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_sdk_error_status_code_429_retries(mock_sleep, mock_client):
    """SDKError with status_code=429 triggers rate limit retry."""
    from mistralai.client.errors import SDKError

    err = SDKError(message="Rate limited", raw_response=MagicMock(status_code=429))
    success = _mock_mistral_response(content="OK")
    mock_client.chat.complete.side_effect = [err, success]

    result = ask_mistral("Hello", response_type=str)

    assert result.output == "OK"
    assert mock_sleep.call_count == 1


# --- SDKError non-rate-limit raises immediately ---


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_sdk_error_non_rate_limit_raises(mock_sleep, mock_client):
    from mistralai.client.errors import SDKError

    err = SDKError(message="Bad request", raw_response=MagicMock(status_code=400))
    mock_client.chat.complete.side_effect = err

    with pytest.raises(SDKError):
        ask_mistral("Hello", response_type=str)

    assert mock_client.chat.complete.call_count == 1
    assert mock_sleep.call_count == 0


# --- usage extraction ---


def test_extract_usage():
    resp = MagicMock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 50
    resp.usage.total_tokens = 150

    usage = _extract_mistral_usage(resp, model="mistral-small-latest")
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.cached_tokens == 0


def test_extract_usage_missing_raises():
    resp = MagicMock()
    resp.usage = None

    with pytest.raises(AttributeError, match="missing usage"):
        _extract_mistral_usage(resp, model="mistral-small-latest")


# --- _parse_wait_time always returns None ---


def test_parse_wait_time_always_none():
    assert _parse_wait_time_from_error(Exception("any error")) is None


# --- verbose ---


def test_set_verbose():
    set_rate_limiter_verbose(True)
    set_rate_limiter_verbose(False)
