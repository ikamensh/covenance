"""Tests for instance-scoped keys and call records."""

from datetime import UTC, datetime, timedelta

from covenance import Covenance, get_default_client
from covenance.record import record_llm_call
from covenance.record import clear_records, get_records
from covenance.usage import TokenUsage


def _make_timestamps(duration_seconds: float) -> tuple[datetime, datetime]:
    ended_at = datetime.now(UTC)
    started_at = ended_at - timedelta(seconds=duration_seconds)
    return started_at, ended_at


def test_default_instance_shares_global_record_store():
    """Property test: default instance mirrors module-level record history."""
    clear_records()
    started_at, ended_at = _make_timestamps(0.4)

    record_llm_call(
        model="gpt-5",
        provider="openai",
        usage=TokenUsage(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            cached_tokens=0,
        ),
        started_at=started_at,
        ended_at=ended_at,
    )

    default_client = get_default_client()
    assert len(default_client.get_records()) == 1
    assert len(get_records()) == 1


def test_instance_records_are_isolated():
    """Property test: instance records are isolated from the default store."""
    clear_records()
    client = Covenance(label="isolation-test")
    started_at, ended_at = _make_timestamps(1.1)

    # Record to the client's store via metrics.record_llm_call
    record_llm_call(
        model="gemini-2.5-flash",
        provider="gemini",
        usage=TokenUsage(
            prompt_tokens=5,
            completion_tokens=7,
            total_tokens=12,
            cached_tokens=0,
        ),
        started_at=started_at,
        ended_at=ended_at,
        record_store=client.get_record_store(),
    )

    assert len(client.get_records()) == 1
    assert len(get_records()) == 0

