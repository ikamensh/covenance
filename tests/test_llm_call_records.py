"""Tests for always-on LLM call logging."""

from datetime import UTC, datetime, timedelta

import pytest

from covenance.record import (
    Record,
    RecordStore,
    TokenUsage,
    clear_records,
    get_llm_call_records_path,
    get_records,
    load_records_from_jsonl,
    print_usage,
    record_llm_call,
    set_records_dir,
    usage_summary,
)


def _make_timestamps(duration_seconds: float) -> tuple[datetime, datetime]:
    ended_at = datetime.now(UTC)
    started_at = ended_at - timedelta(seconds=duration_seconds)
    return started_at, ended_at


def test_record_llm_call_is_always_logged():
    clear_records()
    started_at, ended_at = _make_timestamps(1.2)

    record_llm_call(
        model="gpt-4o",
        provider="openai",
        usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cached_tokens=0,
        ),
        started_at=started_at,
        ended_at=ended_at,
    )

    records = get_records()
    assert len(records) == 1
    assert records[0].model == "gpt-4o"
    assert records[0].duration_seconds == 1.2


def test_record_llm_call_persists_to_dir(tmp_path):
    clear_records()
    set_records_dir(tmp_path)

    try:
        started_at, ended_at = _make_timestamps(2.0)
        record_llm_call(
            model="claude-haiku-4-5",
            provider="anthropic",
            usage=TokenUsage(
                prompt_tokens=5,
                completion_tokens=7,
                total_tokens=12,
                cached_tokens=0,
            ),
            started_at=started_at,
            ended_at=ended_at,
        )

        records_path = get_llm_call_records_path()
        assert records_path is not None
        assert records_path.exists()

        lines = records_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        parsed = Record.model_validate_json(lines[0])
        assert parsed.model == "claude-haiku-4-5"
        assert parsed.provider == "anthropic"
    finally:
        set_records_dir(None)


def _make_record(
    model: str = "gpt-4o",
    provider: str = "openai",
    tokens_in: int = 100,
    tokens_out: int = 50,
    tokens_cached: int = 0,
    cost: float | None = 0.001,
    duration: float = 1.0,
) -> Record:
    """Create a test Record with reasonable defaults."""
    ended = datetime.now(UTC)
    started = ended - timedelta(seconds=duration)
    return Record(
        model=model,
        provider=provider,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        tokens_cached=tokens_cached,
        tokens_total=tokens_in + tokens_out,
        cost_usd=cost,
        duration_seconds=duration,
        started_at=started.isoformat(),
        ended_at=ended.isoformat(),
    )


class TestUsageSummary:
    """Test usage_summary aggregation logic."""

    def test_empty_records_returns_zeros(self):
        """Empty record list produces zero totals."""
        summary = usage_summary([])
        assert summary["calls"] == 0
        assert summary["tokens_total"] == 0
        assert summary["cost_usd"] == 0.0
        assert summary["models"] == set()

    def test_aggregates_multiple_records(self):
        """Tokens and costs sum correctly across records."""
        records = [
            _make_record(tokens_in=100, tokens_out=50, cost=0.001),
            _make_record(tokens_in=200, tokens_out=100, cost=0.002),
        ]
        summary = usage_summary(records)
        assert summary["calls"] == 2
        assert summary["tokens_input"] == 300
        assert summary["tokens_output"] == 150
        assert summary["tokens_total"] == 450
        assert summary["cost_usd"] == pytest.approx(0.003)

    def test_tracks_unique_models(self):
        """Model tracking deduplicates correctly."""
        records = [
            _make_record(model="gpt-4o", provider="openai"),
            _make_record(model="gpt-4o", provider="openai"),  # Duplicate
            _make_record(model="claude-sonnet", provider="anthropic"),
        ]
        summary = usage_summary(records)
        assert summary["models"] == {"openai/gpt-4o", "anthropic/claude-sonnet"}

    def test_handles_none_cost(self):
        """Records with None cost don't break aggregation."""
        records = [
            _make_record(cost=0.001),
            _make_record(cost=None),  # Unknown pricing
        ]
        summary = usage_summary(records)
        assert summary["cost_usd"] == pytest.approx(0.001)

    def test_cached_tokens_tracked(self):
        """Cached tokens are tracked separately."""
        records = [_make_record(tokens_in=100, tokens_cached=30)]
        summary = usage_summary(records)
        assert summary["tokens_cached"] == 30


class TestPrintUsage:
    """Test print_usage output formatting."""

    def test_no_calls_prints_message(self, capsys):
        """Empty records print a simple message."""
        print_usage([])
        captured = capsys.readouterr()
        assert "No LLM calls recorded" in captured.out

    def test_basic_output_format(self, capsys):
        """Standard output includes expected fields."""
        records = [_make_record(tokens_in=100, tokens_out=50, cost=0.05)]
        print_usage(records)
        captured = capsys.readouterr()
        assert "Calls: 1" in captured.out
        assert "Tokens:" in captured.out
        assert "Cost:" in captured.out
        assert "Models:" in captured.out

    def test_cached_tokens_shown_separately(self, capsys):
        """When cached tokens exist, they're shown as 'new + cached'."""
        records = [_make_record(tokens_in=100, tokens_cached=30)]
        print_usage(records)
        captured = capsys.readouterr()
        # Should show "70 new + 30 cached" format
        assert "new" in captured.out
        assert "cached" in captured.out

    def test_cent_format_for_small_cost(self, capsys):
        """cost_format='cent' shows cents for small values."""
        records = [_make_record(cost=0.005)]  # Half a cent
        print_usage(records, cost_format="cent")
        captured = capsys.readouterr()
        assert "¢" in captured.out

    def test_exponential_format_for_small_cost(self, capsys):
        """cost_format='exponential' shows scientific notation."""
        records = [_make_record(cost=0.001)]
        print_usage(records, cost_format="exponential")
        captured = capsys.readouterr()
        assert "e" in captured.out.lower()  # Scientific notation


class TestLoadRecordsFromJsonl:
    """Test JSONL loading and validation."""

    def test_loads_valid_jsonl(self, tmp_path):
        """Valid JSONL file is parsed correctly."""
        records_file = tmp_path / "records.jsonl"
        record = _make_record()
        records_file.write_text(record.model_dump_json() + "\n")

        loaded = load_records_from_jsonl(records_file)
        assert len(loaded) == 1
        assert loaded[0].model == record.model

    def test_skips_blank_lines(self, tmp_path):
        """Blank lines in JSONL don't cause errors."""
        records_file = tmp_path / "records.jsonl"
        record = _make_record()
        content = f"\n{record.model_dump_json()}\n\n"
        records_file.write_text(content)

        loaded = load_records_from_jsonl(records_file)
        assert len(loaded) == 1

    def test_raises_on_missing_file(self, tmp_path):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_records_from_jsonl(tmp_path / "nonexistent.jsonl")

    def test_sorts_by_start_time(self, tmp_path):
        """Records are sorted chronologically."""
        records_file = tmp_path / "records.jsonl"

        # Create records with different timestamps
        now = datetime.now(UTC)
        r1 = Record(
            model="m1",
            provider="p",
            tokens_input=1,
            tokens_output=1,
            tokens_total=2,
            duration_seconds=1.0,
            started_at=(now - timedelta(seconds=10)).isoformat(),
            ended_at=(now - timedelta(seconds=9)).isoformat(),
        )
        r2 = Record(
            model="m2",
            provider="p",
            tokens_input=1,
            tokens_output=1,
            tokens_total=2,
            duration_seconds=1.0,
            started_at=(now - timedelta(seconds=5)).isoformat(),
            ended_at=(now - timedelta(seconds=4)).isoformat(),
        )
        # Write in reverse order
        records_file.write_text(
            r2.model_dump_json() + "\n" + r1.model_dump_json() + "\n"
        )

        loaded = load_records_from_jsonl(records_file)
        assert loaded[0].model == "m1"  # Earlier record comes first
        assert loaded[1].model == "m2"


class TestRecordStore:
    """Test RecordStore class directly."""

    def test_set_records_dir_to_none_disables_persistence(self):
        """Setting records_dir to None disables file persistence."""
        store = RecordStore(records_dir="/tmp/test")
        store.set_llm_call_records_dir(None)
        assert store.get_llm_call_records_dir() is None
        assert store.get_llm_call_records_path() is None

    def test_clear_records_empties_memory(self):
        """clear_records removes in-memory records."""
        store = RecordStore()
        store._records.append(_make_record())
        assert len(store.get_records()) == 1
        store.clear_records()
        assert len(store.get_records()) == 0
