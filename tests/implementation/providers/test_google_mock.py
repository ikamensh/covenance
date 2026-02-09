"""Mock-based tests for google_client.py ask_gemini and usage extraction."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from covenance.clients.google_client import (
    _extract_gemini_usage,
    ask_gemini,
    set_rate_limiter_verbose,
)
from covenance.exceptions import StructuredOutputParsingError


class SampleResponse(BaseModel):
    answer: str
    value: int


def _mock_gemini_response(text=None, parsed=None, usage=None):
    resp = MagicMock()
    resp.text = text
    resp.parsed = parsed
    if usage is None:
        usage = MagicMock()
        usage.prompt_token_count = 10
        usage.candidates_token_count = 5
        usage.total_token_count = 15
        usage.cached_content_token_count = 0
    resp.usage_metadata = usage
    return resp


# --- ask_gemini plain text ---


@patch("covenance.clients.google_client.client")
@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_plain_text(mock_sleep, mock_client):
    resp = _mock_gemini_response(text="Hello, world!")
    mock_client.models.generate_content.return_value = resp

    result = ask_gemini("Hello", response_type=str)

    assert result.output == "Hello, world!"
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args[1]
    assert call_kwargs["config"]["response_mime_type"] == "text/plain"


@patch("covenance.clients.google_client.client")
@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_plain_text_none_response_type(mock_sleep, mock_client):
    resp = _mock_gemini_response(text="Hello")
    mock_client.models.generate_content.return_value = resp

    result = ask_gemini("Hello", response_type=None)
    assert result.output == "Hello"


# --- ask_gemini structured output ---


@patch("covenance.clients.google_client.client")
@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_structured_output(mock_sleep, mock_client):
    parsed = SampleResponse(answer="test", value=42)
    resp = _mock_gemini_response(parsed=parsed)
    mock_client.models.generate_content.return_value = resp

    result = ask_gemini("Hello", response_type=SampleResponse)

    assert result.output.answer == "test"
    assert result.output.value == 42
    call_kwargs = mock_client.models.generate_content.call_args[1]
    assert call_kwargs["config"]["response_mime_type"] == "application/json"
    assert call_kwargs["config"]["response_schema"] is SampleResponse


# --- system message, temperature, client_override ---


@patch("covenance.clients.google_client.client")
@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_with_system_message(mock_sleep, mock_client):
    resp = _mock_gemini_response(text="Response")
    mock_client.models.generate_content.return_value = resp

    ask_gemini("Hello", sys_msg="Be concise", response_type=str)

    cfg = mock_client.models.generate_content.call_args[1]["config"]
    assert cfg["system_instruction"] == "Be concise"


@patch("covenance.clients.google_client.client")
@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_with_temperature(mock_sleep, mock_client):
    resp = _mock_gemini_response(text="Response")
    mock_client.models.generate_content.return_value = resp

    ask_gemini("Hello", temperature=0.5, response_type=str)

    cfg = mock_client.models.generate_content.call_args[1]["config"]
    assert cfg["temperature"] == 0.5


@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_with_client_override(mock_sleep):
    override = MagicMock()
    resp = _mock_gemini_response(text="Response")
    override.models.generate_content.return_value = resp

    result = ask_gemini("Hello", response_type=str, client_override=override)

    assert result.output == "Response"
    override.models.generate_content.assert_called_once()


# --- error paths ---


@patch("covenance.clients.google_client.client")
@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_none_text_raises(mock_sleep, mock_client):
    resp = _mock_gemini_response(text=None)
    mock_client.models.generate_content.return_value = resp

    with pytest.raises(StructuredOutputParsingError, match="text field is None"):
        ask_gemini("Hello", response_type=str)


@patch("covenance.clients.google_client.client")
@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_none_parsed_raises(mock_sleep, mock_client):
    resp = _mock_gemini_response(parsed=None)
    mock_client.models.generate_content.return_value = resp

    with pytest.raises(StructuredOutputParsingError, match="parsed field is None"):
        ask_gemini("Hello", response_type=SampleResponse)


# --- rate limit retry ---


@patch("covenance.clients.google_client.client")
@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_retries_on_rate_limit(mock_sleep, mock_client):
    from google.genai.errors import ClientError

    # ClientError(code, response_json) - string match on "RESOURCE_EXHAUSTED" and "429"
    rate_error = ClientError(
        429, {"error": {"message": "RESOURCE_EXHAUSTED 429. Please retry in 2s."}}
    )
    success_resp = _mock_gemini_response(text="Success")
    mock_client.models.generate_content.side_effect = [rate_error, success_resp]

    result = ask_gemini("Hello", response_type=str)

    assert result.output == "Success"
    assert result.tpm_retries == 1
    assert result.tpm_wait_seconds >= 1.0
    assert mock_sleep.call_count == 1


@patch("covenance.clients.google_client.client")
@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_non_rate_limit_error_raises(mock_sleep, mock_client):
    from google.genai.errors import ClientError

    err = ClientError(400, {"error": {"message": "Bad request"}})
    mock_client.models.generate_content.side_effect = err

    with pytest.raises(ClientError):
        ask_gemini("Hello", response_type=str)

    assert mock_client.models.generate_content.call_count == 1
    assert mock_sleep.call_count == 0


@patch("covenance.clients.google_client.client")
@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_rate_limit_by_error_dict(mock_sleep, mock_client):
    """Detects rate limit from error dict with RESOURCE_EXHAUSTED status."""
    from google.genai.errors import ClientError

    err = ClientError(200, {"error": {"message": "some error"}})
    err.error = {"status": "RESOURCE_EXHAUSTED", "code": 429}
    success_resp = _mock_gemini_response(text="OK")
    mock_client.models.generate_content.side_effect = [err, success_resp]

    result = ask_gemini("Hello", response_type=str)
    assert result.output == "OK"
    assert mock_sleep.call_count == 1


@patch("covenance.clients.google_client.client")
@patch("covenance.clients.google_client.time.sleep", autospec=True)
def test_ask_gemini_rate_limit_by_string_match(mock_sleep, mock_client):
    """Detects rate limit from string containing RESOURCE_EXHAUSTED and 429."""
    from google.genai.errors import ClientError

    err = ClientError(200, {"error": {"message": "RESOURCE_EXHAUSTED 429 too many"}})
    success_resp = _mock_gemini_response(text="OK")
    mock_client.models.generate_content.side_effect = [err, success_resp]

    result = ask_gemini("Hello", response_type=str)
    assert result.output == "OK"


# --- usage extraction ---


def test_extract_gemini_usage():
    resp = MagicMock()
    resp.usage_metadata.prompt_token_count = 100
    resp.usage_metadata.candidates_token_count = 50
    resp.usage_metadata.total_token_count = 150
    resp.usage_metadata.cached_content_token_count = 20

    usage = _extract_gemini_usage(resp, model="gemini-2.5-flash")

    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150
    assert usage.cached_tokens == 20


def test_extract_gemini_usage_no_cache():
    resp = MagicMock()
    resp.usage_metadata.prompt_token_count = 10
    resp.usage_metadata.candidates_token_count = 5
    resp.usage_metadata.total_token_count = 15
    resp.usage_metadata.cached_content_token_count = None

    usage = _extract_gemini_usage(resp, model="gemini-2.5-flash")
    assert usage.cached_tokens == 0


# --- verbose flag ---


def test_set_verbose():
    set_rate_limiter_verbose(True)
    set_rate_limiter_verbose(False)
