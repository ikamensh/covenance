"""Tests for instance-scoped keys and call records."""

from datetime import UTC, datetime, timedelta

from covenance import Covenance
from covenance.record import TokenUsage, clear_records, get_records, record_llm_call


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


def test_provider_routing():
    """Verify model names route to correct providers."""
    client = Covenance()
    cases = [
        ("gpt-5", "openai"),
        ("o3", "openai"),
        ("gemini-2.5-flash", "gemini"),
        ("claude-3.5-sonnet", "anthropic"),
        ("mistral-large", "mistral"),
        ("grok-4", "grok"),
        ("grok-4-fast", "grok"),
        ("grok-3-mini", "grok"),
        ("meta-llama/llama-3-70b", "openrouter"),
    ]
    for model, expected in cases:
        assert client._get_provider(model) == expected, (
            f"{model} should route to {expected}"
        )


def test_explicit_key_triggers_immediate_client_creation():
    """Explicit keys are validated at init by creating the SDK client immediately."""
    client = Covenance(openai_api_key="sk-test-key-for-validation")

    # The OpenAI client should already be instantiated (not lazy)
    openai_lazy = client._clients["openai"]
    assert openai_lazy._client is not None, (
        "Explicit key should trigger immediate client creation"
    )

    # Other providers should still be lazy (no explicit key)
    anthropic_lazy = client._clients["anthropic"]
    assert anthropic_lazy._client is None, "No explicit key means client stays lazy"
