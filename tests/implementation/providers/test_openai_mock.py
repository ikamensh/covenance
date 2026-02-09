"""Mock-based tests for openai_client.py ask_openai_compatible_structured and usage."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from covenance.clients.openai_client import (
    _extract_openai_compatible_usage,
    ask_openai,
    ask_openai_compatible_structured,
    set_rate_limiter_verbose,
)
from covenance.exceptions import StructuredOutputParsingError


class SampleResponse(BaseModel):
    answer: str
    value: int


def _mock_openai_response(output_text=None, output_parsed=None, input_tokens=10, output_tokens=5):
    resp = MagicMock()
    resp.output_text = output_text
    resp.output_parsed = output_parsed
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.usage.total_tokens = input_tokens + output_tokens
    resp.usage.input_tokens_details = None
    return resp


# --- plain text ---


def test_plain_text():
    client = MagicMock()
    resp = _mock_openai_response(output_text="Hello!")
    client.responses.create.return_value = resp

    result = ask_openai_compatible_structured(
        client=client, user_msg="Hello", response_type=str, model="gpt-4o"
    )

    assert result.output == "Hello!"
    client.responses.create.assert_called_once()
    call_kwargs = client.responses.create.call_args[1]
    assert call_kwargs["instructions"] is None
    assert call_kwargs["model"] == "gpt-4o"


def test_plain_text_none_response_type():
    client = MagicMock()
    resp = _mock_openai_response(output_text="Hi")
    client.responses.create.return_value = resp

    result = ask_openai_compatible_structured(
        client=client, user_msg="Hi", response_type=None, model="gpt-4o"
    )
    assert result.output == "Hi"


# --- structured output ---


def test_structured_output():
    client = MagicMock()
    parsed = SampleResponse(answer="Paris", value=95)
    resp = _mock_openai_response(output_parsed=parsed)
    client.responses.parse.return_value = resp

    result = ask_openai_compatible_structured(
        client=client, user_msg="Q", response_type=SampleResponse, model="gpt-4o"
    )

    assert result.output.answer == "Paris"
    client.responses.parse.assert_called_once()
    call_kwargs = client.responses.parse.call_args[1]
    assert call_kwargs["text_format"] is SampleResponse


# --- params forwarding ---


def test_sys_msg_and_temperature():
    client = MagicMock()
    resp = _mock_openai_response(output_text="R")
    client.responses.create.return_value = resp

    ask_openai_compatible_structured(
        client=client, user_msg="Q", response_type=str,
        sys_msg="Be brief", temperature=0.2, model="gpt-4o"
    )

    call_kwargs = client.responses.create.call_args[1]
    assert call_kwargs["instructions"] == "Be brief"
    assert call_kwargs["temperature"] == 0.2


# --- error paths ---


def test_none_output_raises():
    client = MagicMock()
    resp = _mock_openai_response(output_parsed=None)
    client.responses.parse.return_value = resp

    with pytest.raises(StructuredOutputParsingError, match="Empty output"):
        ask_openai_compatible_structured(
            client=client, user_msg="Q", response_type=SampleResponse, model="gpt-4o"
        )


def test_none_text_output_raises():
    client = MagicMock()
    resp = _mock_openai_response(output_text=None)
    client.responses.create.return_value = resp

    with pytest.raises(StructuredOutputParsingError, match="Empty output"):
        ask_openai_compatible_structured(
            client=client, user_msg="Q", response_type=str, model="gpt-4o"
        )


# --- rate limit retry ---


@patch("covenance.clients.openai_client.time.sleep", autospec=True)
def test_retries_on_rate_limit(mock_sleep):
    from openai import RateLimitError

    client = MagicMock()
    err = RateLimitError(
        message="Please try again in 2s",
        response=MagicMock(status_code=429),
        body={},
    )
    success = _mock_openai_response(output_text="OK")
    client.responses.create.side_effect = [err, success]

    result = ask_openai_compatible_structured(
        client=client, user_msg="Q", response_type=str, model="gpt-4o"
    )

    assert result.output == "OK"
    assert result.tpm_retries == 1
    assert mock_sleep.call_count == 1


@patch("covenance.clients.openai_client.time.sleep", autospec=True)
def test_rate_limit_exhausted_raises(mock_sleep):
    from openai import RateLimitError

    client = MagicMock()
    err = RateLimitError(
        message="Rate limit", response=MagicMock(status_code=429), body={}
    )
    client.responses.create.side_effect = err

    with pytest.raises(RateLimitError):
        ask_openai_compatible_structured(
            client=client, user_msg="Q", response_type=str, model="gpt-4o"
        )

    assert client.responses.create.call_count == 100


# --- usage extraction ---


def test_extract_usage():
    resp = MagicMock()
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 50
    resp.usage.total_tokens = 150
    resp.usage.input_tokens_details = None

    usage = _extract_openai_compatible_usage(resp, model="gpt-4o")
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.cached_tokens == 0


def test_extract_usage_with_cached():
    resp = MagicMock()
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 50
    resp.usage.total_tokens = 150
    resp.usage.input_tokens_details = MagicMock()
    resp.usage.input_tokens_details.cached_tokens = 30

    usage = _extract_openai_compatible_usage(resp, model="gpt-4o")
    assert usage.cached_tokens == 30


def test_extract_usage_missing_raises():
    resp = MagicMock()
    resp.usage = None

    with pytest.raises(AttributeError, match="missing usage"):
        _extract_openai_compatible_usage(resp, model="gpt-4o")


def test_extract_usage_no_usage_attr():
    resp = MagicMock(spec=[])  # no attributes

    with pytest.raises(AttributeError, match="missing usage"):
        _extract_openai_compatible_usage(resp, model="gpt-4o")


def test_extract_usage_provider_name_in_error():
    resp = MagicMock()
    resp.usage = None

    with pytest.raises(AttributeError, match="OpenAI"):
        _extract_openai_compatible_usage(resp, model="gpt-4o", provider="openai")

    with pytest.raises(AttributeError, match="Grok"):
        _extract_openai_compatible_usage(resp, model="grok-4", provider="grok")


# --- ask_openai wrapper ---


@patch("covenance.clients.openai_client.client")
@patch("covenance.clients.openai_client.time.sleep", autospec=True)
def test_ask_openai_delegates(mock_sleep, mock_client):
    resp = _mock_openai_response(output_text="Hi")
    mock_client.responses.create.return_value = resp

    result = ask_openai("Hello", response_type=str, model="gpt-4o")
    assert result.output == "Hi"
    mock_client.responses.create.assert_called_once()


@patch("covenance.clients.openai_client.time.sleep", autospec=True)
def test_ask_openai_with_client_override(mock_sleep):
    override = MagicMock()
    resp = _mock_openai_response(output_text="Hi")
    override.responses.create.return_value = resp

    result = ask_openai("Hello", response_type=str, client_override=override)
    assert result.output == "Hi"
    override.responses.create.assert_called_once()


# --- verbose ---


def test_set_verbose():
    set_rate_limiter_verbose(True)
    set_rate_limiter_verbose(False)
