"""Tests for metrics collection integrated with operation lifecycle.

Property tests verifying:
1. start_operation_attempt_by_id() starts a metrics context
2. LLM calls within the context are recorded
3. complete_operation_by_id() / fail_operation_by_id() stop the context and save metrics to DB
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from covenance.metrics import MetricsContext
from covenance.usage import TokenUsage

pytest.importorskip(
    "supa_client",
    reason="Supabase integration tests require the supa_client package.",
)


def _make_timestamps(duration_seconds: float) -> tuple[datetime, datetime]:
    """Helper to create started_at/ended_at pair for tests."""
    ended_at = datetime.now(UTC)
    started_at = ended_at - timedelta(seconds=duration_seconds)
    return started_at, ended_at


from supa_client.supatypes.backend_op import (
    complete_operation_by_id,
    fail_operation_by_id,
    start_operation_attempt_by_id,
)


class MockSelectResponse:
    """Mock Supabase response object."""

    def __init__(self, data: list[dict] | None = None):
        self.data = data or []


class MockUpdateResponse:
    """Mock Supabase update response."""

    def __init__(self, data: dict | None = None):
        self.data = data or {}


def create_mock_supabase_for_operation(operation_data: dict):
    """Create a mock Supabase client configured for operation queries.

    Handles query pattern: select().eq().execute() (for get_backend_operation_by_id)
    """
    client = MagicMock()
    table_mock = MagicMock()
    client.table.return_value = table_mock

    # For select queries by id: select().eq().execute()
    select_mock = MagicMock()
    eq_mock = MagicMock()
    eq_mock.execute.return_value = MockSelectResponse([operation_data])
    select_mock.eq.return_value = eq_mock
    table_mock.select.return_value = select_mock

    # For update queries: update().eq().execute()
    update_mock = MagicMock()
    update_eq_mock = MagicMock()
    update_eq_mock.execute.return_value = MockUpdateResponse()
    update_mock.eq.return_value = update_eq_mock
    table_mock.update.return_value = update_mock

    return client, table_mock


@pytest.fixture
def sample_operation_data():
    """Sample backend operation data from DB."""
    return {
        "id": 123,
        "created_at": datetime(2024, 1, 1, 0, 0, 0),
        "dpia_session_id": "test-session-id",
        "operation_name": "dpia_creation",
        "created_by": "test",
        "params": None,
        "attempts": 0,
        "succeeded": None,
        "duration": None,
        "error_message": None,
        "enqueued": True,
        "metrics": None,
    }


@pytest.fixture(autouse=True)
def reset_metrics_context():
    """Reset metrics context before and after each test."""
    from covenance.metrics import MetricsContext

    MetricsContext.set_current(None)
    yield
    MetricsContext.set_current(None)


def test_start_operation_attempt_creates_metrics_context(sample_operation_data):
    """Property test: start_operation_attempt_by_id creates a metrics context for recording."""
    client, table_mock = create_mock_supabase_for_operation(sample_operation_data)

    # Before: no metrics context
    assert MetricsContext.current() is None

    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        operation = start_operation_attempt_by_id(sample_operation_data["id"])

    # After: metrics context is active
    ctx = MetricsContext.current()
    assert ctx is not None
    assert operation.attempts == 1


def test_metrics_recorded_during_operation(sample_operation_data):
    """Property test: LLM calls between start and complete are recorded."""
    client, table_mock = create_mock_supabase_for_operation(sample_operation_data)

    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        start_operation_attempt_by_id(sample_operation_data["id"])

    # Simulate an LLM call recording
    from covenance.metrics import record_llm_call

    usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cached_tokens=10,
    )
    started_at, ended_at = _make_timestamps(1.5)
    record_llm_call(
        model="gemini-2.5-flash",
        provider="gemini",
        usage=usage,
        started_at=started_at,
        ended_at=ended_at,
    )

    # Verify the call was recorded
    ctx = MetricsContext.current()
    assert ctx is not None
    records = ctx.get_records()
    assert len(records) == 1
    assert records[0].model == "gemini-2.5-flash"
    assert records[0].tokens_input == 100
    assert records[0].tokens_output == 50
    assert records[0].tokens_cached == 10
    assert records[0].duration_seconds == 1.5
    # Verify timestamps are present for sequence reconstruction
    assert records[0].started_at is not None
    assert records[0].ended_at is not None
    assert records[0].started_at < records[0].ended_at


def test_complete_operation_saves_metrics(sample_operation_data):
    """Property test: complete_operation collects and saves metrics to DB."""
    # Use mutable state that gets updated
    operation_state = sample_operation_data.copy()
    client, table_mock = create_mock_supabase_for_operation(operation_state)

    def update_mock_response():
        eq_mock = table_mock.select.return_value.eq.return_value
        eq_mock.execute.return_value = MockSelectResponse([operation_state.copy()])

    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        update_mock_response()
        operation = start_operation_attempt_by_id(operation_state["id"])
        operation_state["attempts"] = operation.attempts

    # Simulate LLM calls
    from covenance.metrics import record_llm_call

    s1, e1 = _make_timestamps(1.0)
    record_llm_call(
        model="gemini-2.5-flash",
        provider="gemini",
        usage=TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150, cached_tokens=0
        ),
        started_at=s1,
        ended_at=e1,
    )
    s2, e2 = _make_timestamps(2.0)
    record_llm_call(
        model="gpt-5-mini",
        provider="openai",
        usage=TokenUsage(
            prompt_tokens=200, completion_tokens=100, total_tokens=300, cached_tokens=50
        ),
        started_at=s2,
        ended_at=e2,
        tpm_retry_wait_seconds=0.5,
    )

    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        update_mock_response()
        complete_operation_by_id(operation_state["id"], duration=5.0)

    # Verify metrics context was stopped
    assert MetricsContext.current() is None

    # Verify update was called (twice: once for start_operation_attempt_by_id, once for complete_operation_by_id)
    assert table_mock.update.call_count == 2

    # Get the second update call (from complete_operation_by_id)
    update_call_args = table_mock.update.call_args_list[1][0][0]

    assert "succeeded" in update_call_args
    assert update_call_args["succeeded"] is True
    assert "duration" in update_call_args
    assert update_call_args["duration"] == 5.0
    assert "metrics" in update_call_args

    # Verify metrics structure
    metrics = update_call_args["metrics"]
    assert "host" in metrics
    assert metrics["host"] == "local"  # Tests run locally
    assert "calls" in metrics
    assert isinstance(metrics["calls"], dict)
    assert "attempt_1" in metrics["calls"]  # Attempt 1
    assert len(metrics["calls"]["attempt_1"]) == 2

    # First call: Gemini
    assert metrics["calls"]["attempt_1"][0]["model"] == "gemini-2.5-flash"
    assert metrics["calls"]["attempt_1"][0]["provider"] == "gemini"
    assert metrics["calls"]["attempt_1"][0]["tokens_input"] == 100
    assert "started_at" in metrics["calls"]["attempt_1"][0]
    assert "ended_at" in metrics["calls"]["attempt_1"][0]
    # Verify timestamps are valid ISO 8601 format (parseable)
    call0_start = datetime.fromisoformat(metrics["calls"]["attempt_1"][0]["started_at"])
    call0_end = datetime.fromisoformat(metrics["calls"]["attempt_1"][0]["ended_at"])
    assert call0_start < call0_end

    # Second call: OpenAI
    assert metrics["calls"]["attempt_1"][1]["model"] == "gpt-5-mini"
    assert metrics["calls"]["attempt_1"][1]["provider"] == "openai"
    assert metrics["calls"]["attempt_1"][1]["tokens_cached"] == 50
    assert "started_at" in metrics["calls"]["attempt_1"][1]
    assert "ended_at" in metrics["calls"]["attempt_1"][1]
    call1_start = datetime.fromisoformat(metrics["calls"]["attempt_1"][1]["started_at"])
    call1_end = datetime.fromisoformat(metrics["calls"]["attempt_1"][1]["ended_at"])
    assert call1_start < call1_end


def test_fail_operation_saves_metrics(sample_operation_data):
    """Property test: fail_operation collects and saves metrics to DB."""
    # Use mutable state that gets updated
    operation_state = sample_operation_data.copy()
    client, table_mock = create_mock_supabase_for_operation(operation_state)

    def update_mock_response():
        eq_mock = table_mock.select.return_value.eq.return_value
        eq_mock.execute.return_value = MockSelectResponse([operation_state.copy()])

    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        update_mock_response()
        operation = start_operation_attempt_by_id(operation_state["id"])
        operation_state["attempts"] = operation.attempts

    # Simulate an LLM call before failure
    from covenance.metrics import record_llm_call

    started_at, ended_at = _make_timestamps(0.8)
    record_llm_call(
        model="claude-haiku-4-5",
        provider="anthropic",
        usage=TokenUsage(
            prompt_tokens=500, completion_tokens=100, total_tokens=600, cached_tokens=0
        ),
        started_at=started_at,
        ended_at=ended_at,
    )

    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        update_mock_response()
        fail_operation_by_id(
            operation_state["id"],
            error_message="Something went wrong",
            duration=3.0,
        )

    # Verify metrics context was stopped
    assert MetricsContext.current() is None

    # Verify update was called (twice: once for start_operation_attempt_by_id, once for fail_operation_by_id)
    assert table_mock.update.call_count == 2

    # Get the second update call (from fail_operation_by_id)
    update_call_args = table_mock.update.call_args_list[1][0][0]

    assert update_call_args["succeeded"] is False
    assert update_call_args["error_message"] == "Something went wrong"
    assert update_call_args["duration"] == 3.0
    assert "metrics" in update_call_args

    # Verify metrics structure
    metrics = update_call_args["metrics"]
    assert "host" in metrics
    assert metrics["host"] == "local"  # Tests run locally
    assert isinstance(metrics["calls"], dict)
    assert "attempt_1" in metrics["calls"]  # Attempt 1
    assert len(metrics["calls"]["attempt_1"]) == 1
    assert metrics["calls"]["attempt_1"][0]["model"] == "claude-haiku-4-5"
    assert metrics["calls"]["attempt_1"][0]["provider"] == "anthropic"


def test_complete_without_start_has_no_metrics(sample_operation_data):
    """Property test: complete_operation_by_id without start_operation_attempt_by_id raises RuntimeError."""
    client, table_mock = create_mock_supabase_for_operation(sample_operation_data)

    # No start_operation_attempt_by_id - directly call complete should raise
    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
        pytest.raises(RuntimeError, match="No metrics context active"),
    ):
        complete_operation_by_id(sample_operation_data["id"], duration=1.0)


def test_multiple_llm_calls_all_recorded(sample_operation_data):
    """Property test: multiple LLM calls from different providers are all recorded."""
    # Use mutable state that gets updated
    operation_state = sample_operation_data.copy()
    client, table_mock = create_mock_supabase_for_operation(operation_state)

    def update_mock_response():
        eq_mock = table_mock.select.return_value.eq.return_value
        eq_mock.execute.return_value = MockSelectResponse([operation_state.copy()])

    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        update_mock_response()
        operation = start_operation_attempt_by_id(operation_state["id"])
        operation_state["attempts"] = operation.attempts

    # Simulate multiple LLM calls
    from covenance.metrics import record_llm_call

    providers = [
        ("gemini-2.5-flash", "gemini"),
        ("gpt-5-mini", "openai"),
        ("claude-haiku-4-5", "anthropic"),
        ("mistral-small-latest", "mistral"),
    ]

    for model, provider in providers:
        started_at, ended_at = _make_timestamps(1.0)
        record_llm_call(
            model=model,
            provider=provider,
            usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cached_tokens=0,
            ),
            started_at=started_at,
            ended_at=ended_at,
        )

    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        update_mock_response()
        complete_operation_by_id(operation_state["id"], duration=10.0)

    # Verify all calls recorded (second update call from complete_operation_by_id)
    assert table_mock.update.call_count == 2
    update_call_args = table_mock.update.call_args_list[1][0][0]
    metrics = update_call_args["metrics"]

    assert "host" in metrics
    assert metrics["host"] == "local"  # Tests run locally
    assert isinstance(metrics["calls"], dict)
    assert "attempt_1" in metrics["calls"]  # Attempt 1
    assert len(metrics["calls"]["attempt_1"]) == 4
    # Flatten all calls to check providers
    all_calls = [
        call for calls_list in metrics["calls"].values() for call in calls_list
    ]
    recorded_providers = {call["provider"] for call in all_calls}
    assert recorded_providers == {"gemini", "openai", "anthropic", "mistral"}


def test_host_identifier_in_metrics(sample_operation_data):
    """Property test: metrics include host identifier (local vs Cloud Run)."""
    import os

    from covenance.metrics import MetricsContext

    original_k_service = os.environ.pop("K_SERVICE", None)
    try:
        # Local environment: no K_SERVICE
        ctx = MetricsContext()
        assert ctx.to_dict()["host"] == "local"

        # Cloud Run environment: with K_SERVICE
        os.environ["K_SERVICE"] = "test-cloudrun-service"
        ctx = MetricsContext()
        assert ctx.to_dict()["host"] == "test-cloudrun-service"
    finally:
        os.environ.pop("K_SERVICE", None)
        if original_k_service is not None:
            os.environ["K_SERVICE"] = original_k_service


def test_metrics_merging_on_retry(sample_operation_data):
    """Property test: metrics from multiple attempts are merged with attempt numbers."""
    # Create a mutable copy of operation data that we can update
    operation_state = sample_operation_data.copy()

    # Create a function that returns current state
    def get_operation_data():
        return operation_state.copy()

    client, table_mock = create_mock_supabase_for_operation(operation_state)

    # Update mock to return current state on each call
    def update_mock_response():
        current_data = get_operation_data()
        eq_mock = table_mock.select.return_value.eq.return_value
        eq_mock.execute.return_value = MockSelectResponse([current_data])

    # First attempt
    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        update_mock_response()
        operation = start_operation_attempt_by_id(operation_state["id"])
        operation_state["attempts"] = operation.attempts

    # Simulate LLM call in first attempt
    from covenance.metrics import record_llm_call

    s1, e1 = _make_timestamps(1.0)
    record_llm_call(
        model="gemini-2.5-flash",
        provider="gemini",
        usage=TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150, cached_tokens=0
        ),
        started_at=s1,
        ended_at=e1,
    )

    # Complete first attempt
    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        update_mock_response()
        complete_operation_by_id(operation_state["id"], duration=5.0)

    # Update operation state with first attempt's results
    first_update_args = table_mock.update.call_args_list[1][0][0]
    operation_state["metrics"] = first_update_args["metrics"]
    operation_state["succeeded"] = first_update_args["succeeded"]
    operation_state["duration"] = first_update_args["duration"]

    # Second attempt (retry)
    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        update_mock_response()
        operation = start_operation_attempt_by_id(operation_state["id"])
        operation_state["attempts"] = operation.attempts

    # Simulate different LLM call in second attempt
    s2, e2 = _make_timestamps(2.0)
    record_llm_call(
        model="gpt-5-mini",
        provider="openai",
        usage=TokenUsage(
            prompt_tokens=200, completion_tokens=100, total_tokens=300, cached_tokens=50
        ),
        started_at=s2,
        ended_at=e2,
    )

    # Complete second attempt
    with (
        patch(
            "autodpia.supa_client.supatypes.backend_op.get_supabase_client",
            return_value=client,
        ),
        patch(
            "autodpia.supa_client.client.get_supabase_client",
            return_value=client,
        ),
    ):
        update_mock_response()
        complete_operation_by_id(operation_state["id"], duration=10.0)

    # Verify metrics were merged
    final_update_args = table_mock.update.call_args_list[-1][0][0]
    merged_metrics = final_update_args["metrics"]

    assert "calls" in merged_metrics
    assert isinstance(merged_metrics["calls"], dict)
    assert "attempt_1" in merged_metrics["calls"]  # First attempt
    assert "attempt_2" in merged_metrics["calls"]  # Second attempt
    assert len(merged_metrics["calls"]["attempt_1"]) == 1  # One call in attempt 1
    assert len(merged_metrics["calls"]["attempt_2"]) == 1  # One call in attempt 2

    # Verify calls are from different attempts
    assert merged_metrics["calls"]["attempt_1"][0]["model"] == "gemini-2.5-flash"
    assert merged_metrics["calls"]["attempt_2"][0]["model"] == "gpt-5-mini"
