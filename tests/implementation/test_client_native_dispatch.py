"""Tests for Covenance._get_native_client and _call_native_provider for all providers."""

from unittest.mock import MagicMock, patch

import pytest

from covenance import Covenance


# --- _get_native_client for all providers ---


def test_get_native_client_openai():
    c = Covenance(openai_api_key="test-key")
    client = c._get_native_client("openai")
    assert client is not None
    assert "openai" in c._native_clients


def test_get_native_client_grok():
    c = Covenance(grok_api_key="test-key")
    client = c._get_native_client("grok")
    assert client is not None
    assert "grok" in c._native_clients


def test_get_native_client_gemini():
    c = Covenance(gemini_api_key="test-key")
    client = c._get_native_client("gemini")
    assert client is not None
    assert "gemini" in c._native_clients


def test_get_native_client_mistral():
    c = Covenance(mistral_api_key="test-key")
    client = c._get_native_client("mistral")
    assert client is not None
    assert "mistral" in c._native_clients


def test_get_native_client_anthropic():
    c = Covenance(anthropic_api_key="test-key")
    client = c._get_native_client("anthropic")
    assert client is not None
    assert "anthropic" in c._native_clients


def test_get_native_client_openrouter():
    c = Covenance(openrouter_api_key="test-key")
    client = c._get_native_client("openrouter")
    assert client is not None
    assert "openrouter" in c._native_clients


def test_get_native_client_unknown_returns_none():
    c = Covenance()
    assert c._get_native_client("unknown_provider") is None


def test_get_native_client_cached():
    """Second call returns the same cached client."""
    c = Covenance(openai_api_key="test-key")
    client1 = c._get_native_client("openai")
    client2 = c._get_native_client("openai")
    assert client1 is client2


# --- _get_api_key for unknown provider ---


def test_get_api_key_unknown_returns_none():
    c = Covenance()
    assert c._get_api_key("unknown_provider") is None


# --- _call_native_provider dispatches correctly ---


def test_call_native_provider_gemini():
    c = Covenance(gemini_api_key="test-key")
    mock_client = MagicMock()
    c._native_clients = {"gemini": mock_client}

    with patch("covenance.client.Covenance._get_native_client", return_value=mock_client):
        with patch("covenance.clients.google_client.ask_gemini") as mock_ask:
            mock_result = MagicMock()
            mock_ask.return_value = mock_result

            result = c._call_native_provider(
                user_msg="Hello",
                model="gemini-2.5-flash",
                provider="gemini",
                llm_type=str,
                sys_msg=None,
                temperature=None,
            )

            mock_ask.assert_called_once()
            assert result is mock_result


def test_call_native_provider_mistral():
    c = Covenance(mistral_api_key="test-key")
    mock_client = MagicMock()
    c._native_clients = {"mistral": mock_client}

    with patch("covenance.client.Covenance._get_native_client", return_value=mock_client):
        with patch("covenance.clients.mistral_client.ask_mistral") as mock_ask:
            mock_result = MagicMock()
            mock_ask.return_value = mock_result

            result = c._call_native_provider(
                user_msg="Hello",
                model="mistral-small-latest",
                provider="mistral",
                llm_type=str,
                sys_msg=None,
                temperature=None,
            )

            mock_ask.assert_called_once()
            assert result is mock_result


def test_call_native_provider_anthropic():
    c = Covenance(anthropic_api_key="test-key")
    mock_client = MagicMock()
    c._native_clients = {"anthropic": mock_client}

    with patch("covenance.client.Covenance._get_native_client", return_value=mock_client):
        with patch("covenance.clients.anthropic_client.ask_anthropic") as mock_ask:
            mock_result = MagicMock()
            mock_ask.return_value = mock_result

            result = c._call_native_provider(
                user_msg="Hello",
                model="claude-haiku-4-5",
                provider="anthropic",
                llm_type=str,
                sys_msg=None,
                temperature=None,
            )

            mock_ask.assert_called_once()
            assert result is mock_result


def test_call_native_provider_openrouter():
    c = Covenance(openrouter_api_key="test-key")
    mock_client = MagicMock()
    c._native_clients = {"openrouter": mock_client}

    with patch("covenance.client.Covenance._get_native_client", return_value=mock_client):
        with patch(
            "covenance.clients.openai_client.ask_openai_compatible_structured"
        ) as mock_ask:
            mock_result = MagicMock()
            mock_ask.return_value = mock_result

            result = c._call_native_provider(
                user_msg="Hello",
                model="meta-llama/llama-3-70b",
                provider="openrouter",
                llm_type=str,
                sys_msg=None,
                temperature=None,
            )

            mock_ask.assert_called_once()
            call_kwargs = mock_ask.call_args[1]
            assert call_kwargs["provider"] == "openrouter"


def test_call_native_provider_unknown_raises():
    c = Covenance()
    with pytest.raises(ValueError, match="No native backend"):
        c._call_native_provider(
            user_msg="Hello",
            model="unknown-model",
            provider="unknown",
            llm_type=str,
            sys_msg=None,
            temperature=None,
        )
