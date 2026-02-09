"""Implementation tests for record persistence, loading, and formatting."""

import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from covenance.record import (
    Record,
    RecordStore,
    TokenUsage,
    load_records_from_jsonl,
    print_usage,
    usage_summary,
)


def _make_record(**overrides) -> Record:
    defaults = dict(
        model="gpt-4.1-nano",
        provider="openai",
        backend="native",
        tokens_input=100,
        tokens_output=50,
        tokens_cached=0,
        tokens_total=150,
        cost_usd=0.001,
        duration_seconds=1.5,
        started_at="2025-01-01T00:00:00",
        ended_at="2025-01-01T00:00:01",
    )
    defaults.update(overrides)
    return Record(**defaults)


def _make_usage(**overrides) -> TokenUsage:
    defaults = dict(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    defaults.update(overrides)
    return TokenUsage(**defaults)


# --- RecordStore persistence ---


def test_persist_record_writes_jsonl(tmp_path):
    store = RecordStore(records_dir=tmp_path, label="test")
    store.record_llm_call(
        model="gpt-4.1-nano",
        provider="openai",
        backend="native",
        usage=_make_usage(),
        started_at=datetime(2025, 1, 1, tzinfo=UTC),
        ended_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
    )
    jsonl_file = tmp_path / "llm_call_records.jsonl"
    assert jsonl_file.exists()
    lines = jsonl_file.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["model"] == "gpt-4.1-nano"
    assert data["backend"] == "native"


def test_persist_creates_directory(tmp_path):
    nested = tmp_path / "deep" / "dir"
    store = RecordStore(records_dir=nested)
    store.record_llm_call(
        model="gpt-4.1-nano",
        provider="openai",
        backend="native",
        usage=_make_usage(),
        started_at=datetime(2025, 1, 1, tzinfo=UTC),
        ended_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
    )
    assert nested.exists()
    assert (nested / "llm_call_records.jsonl").exists()


def test_no_persistence_when_dir_not_set(tmp_path):
    store = RecordStore()
    store.record_llm_call(
        model="gpt-4.1-nano",
        provider="openai",
        backend="native",
        usage=_make_usage(),
        started_at=datetime(2025, 1, 1, tzinfo=UTC),
        ended_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
    )
    assert len(store.get_records()) == 1
    assert store.get_llm_call_records_path() is None


def test_set_records_dir_none_disables():
    store = RecordStore(records_dir="/tmp/test")
    assert store._records_dir is not None
    store.set_llm_call_records_dir(None)
    assert store._records_dir is None


def test_custom_records_filename(tmp_path):
    store = RecordStore(records_dir=tmp_path, records_filename="custom.jsonl")
    store.record_llm_call(
        model="gpt-4.1-nano",
        provider="openai",
        backend="native",
        usage=_make_usage(),
        started_at=datetime(2025, 1, 1, tzinfo=UTC),
        ended_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
    )
    assert (tmp_path / "custom.jsonl").exists()
    assert store.get_llm_call_records_path() == tmp_path / "custom.jsonl"


# --- load_records_from_jsonl ---


def test_load_records_from_jsonl(tmp_path):
    r1 = _make_record(started_at="2025-01-01T00:00:02", ended_at="2025-01-01T00:00:03")
    r2 = _make_record(started_at="2025-01-01T00:00:00", ended_at="2025-01-01T00:00:01")
    jsonl_file = tmp_path / "records.jsonl"
    jsonl_file.write_text(r1.model_dump_json() + "\n" + r2.model_dump_json() + "\n")

    loaded = load_records_from_jsonl(jsonl_file)
    assert len(loaded) == 2
    # Should be sorted by started_at
    assert loaded[0].started_at == "2025-01-01T00:00:00"
    assert loaded[1].started_at == "2025-01-01T00:00:02"


def test_load_records_skips_blank_lines(tmp_path):
    r = _make_record()
    jsonl_file = tmp_path / "records.jsonl"
    jsonl_file.write_text("\n" + r.model_dump_json() + "\n\n")
    loaded = load_records_from_jsonl(jsonl_file)
    assert len(loaded) == 1


def test_load_records_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_records_from_jsonl("/nonexistent/path.jsonl")


# --- usage_summary ---


def test_usage_summary_empty():
    s = usage_summary(records=[])
    assert s["calls"] == 0
    assert s["cost_usd"] == 0.0
    assert s["tokens_total"] == 0
    assert s["models"] == set()
    assert s["has_openrouter"] is False


def test_usage_summary_aggregates():
    records = [
        _make_record(
            tokens_input=100, tokens_output=50, cost_usd=0.01,
            tpm_retries=1, structured_output_retries=2,
        ),
        _make_record(
            model="gemini-2.5-flash", provider="gemini", backend="pydantic",
            tokens_input=200, tokens_output=100, tokens_cached=50, cost_usd=0.02,
        ),
    ]
    s = usage_summary(records=records)
    assert s["calls"] == 2
    assert s["tokens_input"] == 300
    assert s["tokens_output"] == 150
    assert s["tokens_cached"] == 50
    assert s["tokens_total"] == 450
    assert s["cost_usd"] == pytest.approx(0.03)
    assert s["tpm_retries"] == 1
    assert s["structured_output_retries"] == 2
    assert {"openai/gpt-4.1-nano", "gemini/gemini-2.5-flash"} == s["models"]
    assert {"native", "pydantic"} == s["backends"]


def test_usage_summary_openrouter_flag():
    records = [
        _make_record(provider="openrouter", model="org/model", cost_usd=None),
    ]
    s = usage_summary(records=records)
    assert s["has_openrouter"] is True
    assert s["cost_usd"] == 0.0  # None cost doesn't add


# --- print_usage formatting ---


def test_print_usage_no_calls(capsys):
    print_usage(records=[])
    output = capsys.readouterr().out
    assert "No LLM calls recorded" in output


def test_print_usage_cent_format(capsys):
    records = [_make_record(cost_usd=0.005)]
    print_usage(records=records, cost_format="cent")
    output = capsys.readouterr().out
    assert "¢" in output


def test_print_usage_exponential_format(capsys):
    records = [_make_record(cost_usd=0.005)]
    print_usage(records=records, cost_format="exponential")
    output = capsys.readouterr().out
    assert "e" in output


def test_print_usage_plain_small_cost(capsys):
    records = [_make_record(cost_usd=0.0012)]
    print_usage(records=records, cost_format="plain")
    output = capsys.readouterr().out
    assert "$0.0012" in output


def test_print_usage_plain_large_cost(capsys):
    records = [_make_record(cost_usd=1.50)]
    print_usage(records=records, cost_format="plain")
    output = capsys.readouterr().out
    assert "$1.50" in output


def test_print_usage_cached_tokens(capsys):
    records = [_make_record(tokens_cached=30)]
    print_usage(records=records)
    output = capsys.readouterr().out
    assert "cached" in output


def test_print_usage_openrouter_note(capsys):
    records = [_make_record(provider="openrouter", model="org/model", cost_usd=None)]
    print_usage(records=records)
    output = capsys.readouterr().out
    assert "excluding OpenRouter" in output


def test_print_usage_show_retries(capsys):
    records = [_make_record(tpm_retries=3, structured_output_retries=1)]
    print_usage(records=records, show_retries=True)
    output = capsys.readouterr().out
    assert "TPM=3" in output
    assert "SO=1" in output


def test_print_usage_backends_shown(capsys):
    records = [_make_record(backend="native")]
    print_usage(records=records)
    output = capsys.readouterr().out
    assert "Backends:" in output
    assert "native" in output


# --- RecordStore caller info ---


def test_record_stores_caller_info():
    store = RecordStore()
    record = store.record_llm_call(
        model="gpt-4.1-nano",
        provider="openai",
        backend="native",
        usage=_make_usage(),
        started_at=datetime(2025, 1, 1, tzinfo=UTC),
        ended_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        caller_function="test_fn",
        caller_file="test_file.py",
        caller_line=42,
    )
    assert record.caller_function == "test_fn"
    assert record.caller_file == "test_file.py"
    assert record.caller_line == 42
