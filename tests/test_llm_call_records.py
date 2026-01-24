"""Tests for always-on LLM call logging."""

from datetime import UTC, datetime, timedelta

from covenance.llm_calls import (
    Record,
    clear_records,
    get_records,
    get_llm_call_records_path,
    set_llm_call_records_dir,
)
from covenance.metrics import LLMOperationContext, record_llm_call
from covenance.usage import TokenUsage


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


def test_record_llm_call_captured_in_context():
    clear_records()
    started_at, ended_at = _make_timestamps(0.5)

    with LLMOperationContext.start(task="test-context") as ctx:
        record_llm_call(
            model="gemini-2.5-flash",
            provider="gemini",
            usage=TokenUsage(
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
                cached_tokens=0,
            ),
            started_at=started_at,
            ended_at=ended_at,
        )
        ctx_records = ctx.get_records()

    assert len(ctx_records) == 1
    assert ctx_records[0].provider == "gemini"
    assert len(get_records()) == 1


def test_record_llm_call_persists_to_dir(tmp_path):
    clear_records()
    set_llm_call_records_dir(tmp_path)

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
        set_llm_call_records_dir(None)

