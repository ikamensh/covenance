"""Tests for openai_client.py - error parsing and usage extraction."""

import pytest

from covenance.clients.openai_client import _parse_wait_time_from_error


class TestParseWaitTimeFromError:
    """Test wait time extraction from OpenAI rate limit errors."""

    def test_parses_decimal_seconds(self):
        """Parse 'Please try again in X.XXXs' pattern."""
        err = Exception("Rate limit exceeded. Please try again in 6.191s")
        assert _parse_wait_time_from_error(err) == pytest.approx(6.191)

    def test_parses_integer_seconds(self):
        """Parse integer seconds without decimal."""
        err = Exception("Please try again in 30s")
        assert _parse_wait_time_from_error(err) == pytest.approx(30.0)

    def test_minimum_wait_enforced(self):
        """Very small wait times are clamped to 0.1s."""
        err = Exception("Please try again in 0.05s")
        assert _parse_wait_time_from_error(err) >= 0.1

    def test_fallback_on_unparseable(self):
        """Return 1.0s default when pattern not found."""
        err = Exception("Unknown error")
        assert _parse_wait_time_from_error(err) == 1.0

    def test_extracts_from_longer_message(self):
        """Pattern found in longer error message."""
        err = Exception(
            "Error code: 429 - You exceeded your current quota. "
            "Please try again in 2.5s or visit billing."
        )
        assert _parse_wait_time_from_error(err) == pytest.approx(2.5)
