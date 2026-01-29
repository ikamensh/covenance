"""Tests for visual.py - model name shortening and timeline formatting."""

from datetime import UTC, datetime, timedelta

import pytest

from covenance.record import Record
from covenance.visual import _format_duration, _shorten_model_name, print_call_timeline


class TestShortenModelName:
    """Test model name abbreviation logic - the function progressively shortens names."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            # Provider prefix shortening (prefix replaced, no extra hyphen)
            ("gemini-2.5-flash", "g2.5-flash"),
            ("claude-sonnet-4", "csonnet-4"),  # claude- → c (no hyphen)
            ("mistral-large", "mi-large"),
            ("codestral-latest", "co-latest"),
            # Providers that stay as-is (short already)
            ("gpt-4o", "gpt-4o"),
            ("grok-beta", "grok-beta"),
            # Date suffix removal (8-digit format only)
            ("claude-sonnet-4-20250514", "csonnet-4"),
            ("gpt-4o-20240513", "gpt-4o"),  # 8-digit date removed
        ],
    )
    def test_basic_shortening(self, model: str, expected: str):
        """Provider prefixes and date suffixes are handled."""
        assert _shorten_model_name(model) == expected

    def test_progressive_shortening_size_suffix(self):
        """Size suffixes (-lite, -mini) are abbreviated when name is too long."""
        # "gemini-2.5-flash-lite" → "g2.5-flash-lite" (14 chars, > max_len=13)
        # → "g2.5-flash-l" (12 chars)
        result = _shorten_model_name("gemini-2.5-flash-lite", max_len=13)
        assert "-l" in result  # -lite → -l
        assert len(result) <= 13

    def test_progressive_shortening_variant_suffix(self):
        """Variant suffixes are abbreviated as last resort."""
        # Force variant shortening with a very short max_len
        result = _shorten_model_name("claude-sonnet-4", max_len=8)
        assert "-s" in result or "…" in result  # -sonnet → -s, or truncated

    def test_truncation_with_ellipsis(self):
        """Very long names get truncated with ellipsis."""
        result = _shorten_model_name("some-very-long-model-name-here", max_len=10)
        assert len(result) == 10
        assert result.endswith("…")

    def test_already_short_unchanged(self):
        """Short names without known prefixes stay unchanged."""
        assert _shorten_model_name("o1-mini", max_len=13) == "o1-mini"


class TestFormatDuration:
    """Test duration formatting - seconds to human-readable."""

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0.5, "0.5s"),
            (1.0, "1.0s"),
            (45.3, "45.3s"),
            (59.9, "59.9s"),
            (60, "1m"),
            (90, "1m30s"),
            (125, "2m5s"),
            (300, "5m"),
            (3600, "60m"),  # No hour format, just large minutes
        ],
    )
    def test_format_duration(self, seconds: float, expected: str):
        """Duration formatting produces expected strings."""
        assert _format_duration(seconds) == expected


def _make_timeline_record(
    model: str = "gpt-4o",
    duration: float = 1.0,
    start_offset: float = 0.0,
) -> Record:
    """Create a Record for timeline testing with specified timing."""
    base = datetime.now(UTC) - timedelta(seconds=10)  # Base time in the past
    started = base + timedelta(seconds=start_offset)
    ended = started + timedelta(seconds=duration)
    return Record(
        model=model,
        provider="test",
        tokens_input=100,
        tokens_output=50,
        tokens_total=150,
        duration_seconds=duration,
        started_at=started.isoformat(),
        ended_at=ended.isoformat(),
    )


class TestPrintCallTimeline:
    """Test the visual timeline display."""

    def test_empty_records_prints_message(self, capsys):
        """Empty list prints 'no calls' message."""
        print_call_timeline([])
        captured = capsys.readouterr()
        assert "No LLM calls" in captured.out

    def test_single_call_shows_bar(self, capsys):
        """Single call shows model name and bar."""
        records = [_make_timeline_record(model="gpt-4o", duration=2.0)]
        print_call_timeline(records)
        captured = capsys.readouterr()
        assert "gpt-4o" in captured.out
        assert "█" in captured.out  # Bar character
        assert "1 calls" in captured.out

    def test_multiple_calls_sorted_by_start(self, capsys):
        """Multiple calls appear in chronological order."""
        records = [
            _make_timeline_record(model="model-b", start_offset=2.0),
            _make_timeline_record(model="model-a", start_offset=0.0),
        ]
        print_call_timeline(records)
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        # Find lines with model names
        model_lines = [l for l in lines if "model-a" in l or "model-b" in l]
        assert len(model_lines) == 2
        # model-a started earlier, should appear first
        assert "model-a" in model_lines[0]
        assert "model-b" in model_lines[1]

    def test_shows_total_duration(self, capsys):
        """Header shows total time span."""
        records = [_make_timeline_record(duration=5.0)]
        print_call_timeline(records)
        captured = capsys.readouterr()
        assert "5.0s" in captured.out

    def test_handles_instant_call(self, capsys):
        """Calls with zero duration don't cause division by zero."""
        records = [_make_timeline_record(duration=0.0)]
        print_call_timeline(records)
        # Should not raise, just display something reasonable
        captured = capsys.readouterr()
        assert "█" in captured.out
