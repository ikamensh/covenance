"""Backend contract tests for record and record store APIs."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from covenance.record import (
    Record,
    RecordStore,
    TokenUsage,
    load_records_from_jsonl,
    print_usage,
    usage_summary,
)


def _make_record(*, backend: str = "native") -> Record:
    ended = datetime.now(UTC)
    started = ended - timedelta(seconds=1)
    return Record(
        model="gpt-4o",
        provider="openai",
        backend=backend,
        tokens_input=10,
        tokens_output=5,
        tokens_total=15,
        duration_seconds=1.0,
        started_at=started.isoformat(),
        ended_at=ended.isoformat(),
    )


class TestRecordBackendContract:
    """Validation-level invariants for backend field."""

    def test_record_requires_backend(self):
        """Record creation without backend fails validation."""
        ended = datetime.now(UTC)
        started = ended - timedelta(seconds=1)

        with pytest.raises(ValidationError, match="backend"):
            Record(
                model="gpt-4o",
                provider="openai",
                tokens_input=10,
                tokens_output=5,
                tokens_total=15,
                duration_seconds=1.0,
                started_at=started.isoformat(),
                ended_at=ended.isoformat(),
            )

    @pytest.mark.parametrize("backend", ["", "invalid", "NATIVE", "none"])
    def test_record_rejects_invalid_backend_values(self, backend: str):
        """Only native/pydantic backend values are accepted."""
        ended = datetime.now(UTC)
        started = ended - timedelta(seconds=1)

        with pytest.raises(ValidationError, match="backend"):
            Record(
                model="gpt-4o",
                provider="openai",
                backend=backend,
                tokens_input=10,
                tokens_output=5,
                tokens_total=15,
                duration_seconds=1.0,
                started_at=started.isoformat(),
                ended_at=ended.isoformat(),
            )


class TestRecordStoreBackendContract:
    """RecordStore should enforce backend at call and summary layers."""

    def test_record_llm_call_requires_backend_argument(self):
        """Backend is a required keyword argument for record_llm_call."""
        store = RecordStore()
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cached_tokens=0,
        )
        ended = datetime.now(UTC)
        started = ended - timedelta(seconds=1)

        with pytest.raises(TypeError):
            store.record_llm_call(
                model="gpt-4o",
                provider="openai",
                usage=usage,
                started_at=started,
                ended_at=ended,
            )

    def test_usage_summary_collects_backend_set(self):
        """Summary includes all used backends without duplicates."""
        records = [_make_record(backend="native"), _make_record(backend="pydantic")]

        summary = usage_summary(records)

        assert summary["backends"] == {"native", "pydantic"}

    def test_print_usage_includes_sorted_backends(self, capsys):
        """Printed summary includes deterministic backend ordering."""
        records = [_make_record(backend="pydantic"), _make_record(backend="native")]

        print_usage(records)
        output = capsys.readouterr().out

        assert "Backends: native, pydantic" in output


class TestJsonlBackendContract:
    """JSONL parsing should reject records without required backend."""

    def test_load_records_from_jsonl_rejects_missing_backend(self, tmp_path):
        """Older records without backend are rejected during load."""
        records_file = tmp_path / "records.jsonl"
        ended = datetime.now(UTC)
        started = ended - timedelta(seconds=1)
        records_file.write_text(
            json.dumps(
                {
                    "model": "gpt-4o",
                    "provider": "openai",
                    "tokens_input": 10,
                    "tokens_output": 5,
                    "tokens_total": 15,
                    "duration_seconds": 1.0,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="backend"):
            load_records_from_jsonl(records_file)
