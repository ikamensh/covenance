"""Tests for dev_tools.py - developer utilities."""

from unittest.mock import patch

from covenance.clients import (
    anthropic_client,
    google_client,
    mistral_client,
    openai_client,
)
from covenance.dev_tools import set_rate_limiter_verbose


def test_set_rate_limiter_verbose_calls_all_providers():
    """set_rate_limiter_verbose forwards to all provider modules."""
    with (
        patch.object(
            anthropic_client, "set_rate_limiter_verbose", autospec=True
        ) as mock_anthropic,
        patch.object(
            google_client, "set_rate_limiter_verbose", autospec=True
        ) as mock_google,
        patch.object(
            openai_client, "set_rate_limiter_verbose", autospec=True
        ) as mock_openai,
        patch.object(
            mistral_client, "set_rate_limiter_verbose", autospec=True
        ) as mock_mistral,
    ):
        set_rate_limiter_verbose(True)

        mock_anthropic.assert_called_once_with(True)
        mock_google.assert_called_once_with(True)
        mock_openai.assert_called_once_with(True)
        mock_mistral.assert_called_once_with(True)


def test_set_rate_limiter_verbose_false():
    """set_rate_limiter_verbose(False) disables verbose mode."""
    with (
        patch.object(
            anthropic_client, "set_rate_limiter_verbose", autospec=True
        ) as mock_anthropic,
        patch.object(
            google_client, "set_rate_limiter_verbose", autospec=True
        ) as mock_google,
        patch.object(
            openai_client, "set_rate_limiter_verbose", autospec=True
        ) as mock_openai,
        patch.object(
            mistral_client, "set_rate_limiter_verbose", autospec=True
        ) as mock_mistral,
    ):
        set_rate_limiter_verbose(False)

        mock_anthropic.assert_called_once_with(False)
        mock_google.assert_called_once_with(False)
        mock_openai.assert_called_once_with(False)
        mock_mistral.assert_called_once_with(False)
