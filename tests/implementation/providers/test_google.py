"""Tests for google_client.py - error parsing and retry logic."""

import pytest

from covenance.clients.google_client import _parse_wait_time_from_error


class MockError:
    """Mock ClientError for testing error parsing."""

    def __init__(self, message: str, error_dict: dict | None = None):
        self._message = message
        self.error = error_dict

    def __str__(self):
        return self._message


class TestParseWaitTimeFromError:
    """Test wait time extraction from Gemini rate limit errors."""

    def test_parses_retry_in_seconds_pattern(self):
        """Parse 'Please retry in X.XXXs.' from error message."""
        err = MockError("Rate limited. Please retry in 51.225s. More info...")
        assert _parse_wait_time_from_error(err) == pytest.approx(51.225)

    def test_parses_integer_seconds(self):
        """Parse integer seconds without decimal."""
        err = MockError("Please retry in 30s.")
        assert _parse_wait_time_from_error(err) == pytest.approx(30.0)

    def test_minimum_wait_time_enforced(self):
        """Very small wait times are clamped to 0.1s minimum."""
        err = MockError("Please retry in 0.01s.")
        assert _parse_wait_time_from_error(err) >= 0.1

    def test_parses_from_error_details_retry_info(self):
        """Extract retryDelay from structured error details."""
        err = MockError(
            "Some error",
            error_dict={
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "45s",
                    }
                ]
            },
        )
        assert _parse_wait_time_from_error(err) == pytest.approx(45.0)

    def test_parses_decimal_from_retry_info(self):
        """Parse decimal seconds from retryDelay."""
        err = MockError(
            "Error",
            error_dict={
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "12.5s",
                    }
                ]
            },
        )
        assert _parse_wait_time_from_error(err) == pytest.approx(12.5)

    def test_fallback_to_default_when_unparseable(self):
        """Return 1.0s default when no wait time can be parsed."""
        err = MockError("Some unrelated error message")
        assert _parse_wait_time_from_error(err) == 1.0

    def test_message_pattern_takes_priority(self):
        """Message pattern is tried before error details."""
        # Both sources present, message pattern should be used first
        err = MockError(
            "Please retry in 5s.",
            error_dict={
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "99s",
                    }
                ]
            },
        )
        assert _parse_wait_time_from_error(err) == pytest.approx(5.0)

    def test_handles_missing_error_attribute(self):
        """Gracefully handles errors without .error attribute."""

        class SimpleError(Exception):
            pass

        err = SimpleError("No details here")
        assert _parse_wait_time_from_error(err) == 1.0

    def test_handles_non_dict_error_attribute(self):
        """Gracefully handles .error that isn't a dict."""
        err = MockError("Message")
        err.error = "not a dict"  # type: ignore
        assert _parse_wait_time_from_error(err) == 1.0

    def test_handles_empty_details_list(self):
        """Empty details list falls back to default."""
        err = MockError("Message", error_dict={"details": []})
        assert _parse_wait_time_from_error(err) == 1.0
