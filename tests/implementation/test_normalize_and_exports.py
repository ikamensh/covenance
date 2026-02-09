"""Tests for model normalization, package exports, and visual helpers."""

from enum import StrEnum
from unittest.mock import patch

from covenance import __all__, __version__
from covenance.client import _normalize_model
from covenance.visual import _shorten_model_name


# --- _normalize_model ---


class FakeModel(StrEnum):
    nano = "gpt-4.1-nano"
    flash = "gemini-2.5-flash"


def test_normalize_model_string_passthrough():
    assert _normalize_model("gpt-4.1-nano") == "gpt-4.1-nano"


def test_normalize_model_enum():
    assert _normalize_model(FakeModel.nano) == "gpt-4.1-nano"
    assert _normalize_model(FakeModel.flash) == "gemini-2.5-flash"


# --- __init__ exports ---


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__ != "0.0.0"  # Should load actual version


def test_all_exports_importable():
    """Every name in __all__ is importable from covenance."""
    import covenance

    for name in __all__:
        assert hasattr(covenance, name), f"{name} listed in __all__ but not importable"


def test_key_exports_are_correct_types():
    import covenance

    assert callable(covenance.ask_llm)
    assert callable(covenance.llm_consensus)
    assert callable(covenance.get_records)
    assert callable(covenance.clear_records)
    assert callable(covenance.print_usage)
    assert callable(covenance.print_call_timeline)
    assert callable(covenance.load_records_from_jsonl)


# --- _shorten_model_name ---


def test_shorten_gemini():
    assert _shorten_model_name("gemini-2.5-flash-lite") == "g2.5-flash-l"


def test_shorten_claude_with_date():
    result = _shorten_model_name("claude-sonnet-4-20250514")
    assert result == "csonnet-4"
    assert len(result) <= 13


def test_shorten_gpt_unchanged():
    result = _shorten_model_name("gpt-4.1-nano")
    assert result == "gpt-4.1-nano"


def test_shorten_very_long_name_truncates():
    result = _shorten_model_name("some-very-long-model-name-that-exceeds-limit", max_len=10)
    assert len(result) <= 10
    assert result.endswith("…")


def test_shorten_mistral():
    result = _shorten_model_name("mistral-small-latest")
    assert len(result) <= 13
    assert result.startswith("mi-")
