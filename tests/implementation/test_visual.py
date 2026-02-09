"""Implementation tests for visual.py call timeline."""

from covenance.record import Record
from covenance.visual import print_call_timeline


def _make_record(model="gpt-4.1-nano", backend="native", start="2025-01-01T00:00:00", end="2025-01-01T00:00:01", **kw) -> Record:
    defaults = dict(
        model=model,
        provider="openai",
        backend=backend,
        tokens_input=10,
        tokens_output=5,
        tokens_total=15,
        cost_usd=0.001,
        duration_seconds=1.0,
        started_at=start,
        ended_at=end,
    )
    defaults.update(kw)
    return Record(**defaults)


def test_timeline_no_calls(capsys):
    print_call_timeline(records=[])
    output = capsys.readouterr().out
    assert "No LLM calls" in output


def test_timeline_single_call(capsys):
    records = [_make_record()]
    print_call_timeline(records=records)
    output = capsys.readouterr().out
    assert "LLM Call Timeline" in output
    assert "1 calls" in output
    assert "█" in output


def test_timeline_shows_backend_tag(capsys):
    records = [
        _make_record(backend="native"),
        _make_record(model="gemini-2.5-flash", backend="pydantic",
                     start="2025-01-01T00:00:01", end="2025-01-01T00:00:02",
                     provider="gemini"),
    ]
    print_call_timeline(records=records)
    output = capsys.readouterr().out
    assert "(N)" in output
    assert "(P)" in output


def test_timeline_parallel_calls(capsys):
    """Overlapping calls should both appear."""
    records = [
        _make_record(start="2025-01-01T00:00:00", end="2025-01-01T00:00:02"),
        _make_record(model="gemini-2.5-flash", provider="gemini", backend="pydantic",
                     start="2025-01-01T00:00:00", end="2025-01-01T00:00:02"),
    ]
    print_call_timeline(records=records)
    output = capsys.readouterr().out
    assert "2 calls" in output
    lines = [l for l in output.split("\n") if "█" in l]
    assert len(lines) == 2


def test_timeline_instant_call(capsys):
    """Call with zero duration should not crash."""
    records = [_make_record(
        start="2025-01-01T00:00:00",
        end="2025-01-01T00:00:00",
        duration_seconds=0.0,
    )]
    print_call_timeline(records=records)
    output = capsys.readouterr().out
    assert "█" in output
