"""Tests for exceptions module."""

import pytest

from covenance.exceptions import (
    MissingProviderError,
    StructuredOutputParsingError,
    require_provider,
)
from covenance.record import TokenUsage


def test_structured_output_parsing_error_carries_usage():
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    err = StructuredOutputParsingError("Parse failed", usage=usage)
    assert err.usage is usage
    assert "Parse failed" in str(err)


def test_structured_output_parsing_error_none_usage():
    err = StructuredOutputParsingError("No usage")
    assert err.usage is None


def test_missing_provider_error_is_import_error():
    assert issubclass(MissingProviderError, ImportError)


def test_require_provider_openai_succeeds():
    require_provider("openai")


def test_require_provider_anthropic_succeeds():
    require_provider("anthropic")


def test_require_provider_google_succeeds():
    require_provider("google")


def test_require_provider_mistral_succeeds():
    require_provider("mistral")


def test_require_provider_unknown_passes():
    """Unknown providers don't have explicit import checks, so they pass."""
    require_provider("unknown_provider")
