"""Additional mock tests for mistral_client.py covering uncovered branches."""

import json
from unittest.mock import MagicMock, patch

import pytest
from mistralai.client.errors import SDKError

from covenance.clients.mistral_client import ask_mistral
from covenance.exceptions import StructuredOutputParsingError


def _mock_mistral_success(content="OK", parsed=None):
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


# --- JSON decode retry path ---


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_json_decode_error_retries_then_succeeds(mock_sleep, mock_client):
    """JSONDecodeError triggers retry, success on second attempt."""
    success_resp = _mock_mistral_success(content="OK")
    mock_client.chat.complete.side_effect = [
        json.JSONDecodeError("bad json", "", 0),
        success_resp,
    ]

    result = ask_mistral("Hello", response_type=str)

    assert result.output == "OK"
    assert mock_client.chat.complete.call_count == 2
    assert mock_sleep.call_count == 1  # 0.5s sleep between retries


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_json_decode_error_exhausted_raises_structured(mock_sleep, mock_client):
    """JSONDecodeError after max retries raises StructuredOutputParsingError."""
    mock_client.chat.complete.side_effect = json.JSONDecodeError("bad", "", 0)

    with pytest.raises(StructuredOutputParsingError, match="invalid JSON"):
        ask_mistral("Hello", response_type=str)

    # 3 JSON parse retries = 3 attempts that trigger JSONDecodeError before raising
    assert mock_client.chat.complete.call_count == 3


# --- catch-all Exception rate limit detection ---


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_unexpected_rate_limit_error_retries(mock_sleep, mock_client):
    """Generic Exception with 429 in message triggers retry."""
    success_resp = _mock_mistral_success(content="OK")
    mock_client.chat.complete.side_effect = [
        Exception("Connection error 429 rate limit"),
        success_resp,
    ]

    result = ask_mistral("Hello", response_type=str)

    assert result.output == "OK"
    assert mock_client.chat.complete.call_count == 2
    assert mock_sleep.call_count == 1


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_unexpected_non_rate_limit_error_raises(mock_sleep, mock_client):
    """Generic Exception without rate limit indicators raises immediately."""
    mock_client.chat.complete.side_effect = Exception("Some random error")

    with pytest.raises(Exception, match="random error"):
        ask_mistral("Hello", response_type=str)

    assert mock_client.chat.complete.call_count == 1
    assert mock_sleep.call_count == 0


# --- SDKError with string-based rate limit detection ---


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_sdk_error_rate_limit_by_string(mock_sleep, mock_client):
    """SDKError with '429' in message retries even without status_code."""

    class MockSDK(SDKError):
        def __init__(self):
            super().__init__(message="429 rate limit exceeded", raw_response=MagicMock(status_code=429))

    success_resp = _mock_mistral_success(content="OK")
    mock_client.chat.complete.side_effect = [MockSDK(), success_resp]

    result = ask_mistral("Hello", response_type=str)
    assert result.output == "OK"


# --- temperature parameter ---


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_temperature_passed(mock_sleep, mock_client):
    resp = _mock_mistral_success(content="OK")
    mock_client.chat.complete.return_value = resp

    ask_mistral("Hello", response_type=str, temperature=0.7)

    call_kwargs = mock_client.chat.complete.call_args[1]
    assert call_kwargs["temperature"] == 0.7


@patch("covenance.clients.mistral_client.client")
@patch("covenance.clients.mistral_client.time.sleep", autospec=True)
def test_temperature_none_passed(mock_sleep, mock_client):
    resp = _mock_mistral_success(content="OK")
    mock_client.chat.complete.return_value = resp

    ask_mistral("Hello", response_type=str, temperature=None)

    call_kwargs = mock_client.chat.complete.call_args[1]
    assert call_kwargs["temperature"] is None
