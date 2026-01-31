"""Tests for the Covenance client public API.

Tests the public interface of the Covenance class without depending on
internal implementation details like _clients dict or lazy client instances.
"""

from datetime import UTC, datetime

import covenance
from covenance import Covenance
from covenance.client import _default_client
from covenance.record import TokenUsage, clear_records, get_records


def _make_timestamps(duration_seconds: float) -> tuple[datetime, datetime]:
    ended_at = datetime.now(UTC)
    started_at = ended_at - (datetime.now(UTC) - datetime.now(UTC).replace(second=0))
    # Simplified: just offset from now
    from datetime import timedelta

    ended_at = datetime.now(UTC)
    started_at = ended_at - timedelta(seconds=duration_seconds)
    return started_at, ended_at


class TestClientRecordStores:
    """Tests for record store isolation between clients."""

    def test_default_instance_shares_global_record_store(self):
        """Default client's records are accessible via module-level get_records()."""
        clear_records()
        started_at, ended_at = _make_timestamps(0.4)

        _default_client.get_record_store().record_llm_call(
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

    def test_instance_records_are_isolated(self):
        """Named client instances have isolated record stores."""
        clear_records()
        client = Covenance(label="isolation-test")
        started_at, ended_at = _make_timestamps(1.1)

        # Record to the client's store directly
        client.get_record_store().record_llm_call(
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
        )

        assert len(client.get_records()) == 1
        assert len(get_records()) == 0


class TestAllClientsAggregation:
    """Tests for aggregating records across multiple Covenance clients."""

    def _add_mock_record(
        self, client: covenance.Covenance, model: str, provider: str, tokens: int = 100
    ) -> None:
        """Add a mock record to a client's record store."""
        usage = TokenUsage(
            prompt_tokens=tokens,
            completion_tokens=tokens // 2,
            total_tokens=tokens + tokens // 2,
        )
        client._record_store.record_llm_call(
            model=model,
            provider=provider,
            usage=usage,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
        )

    def test_get_all_clients_tracks_created_clients(self, clean_client_registry):
        """All created Covenance instances are tracked in the registry."""
        initial_count = len(covenance.get_all_clients())

        client_a = covenance.Covenance(label="test-A")
        client_b = covenance.Covenance(label="test-B")

        all_clients = covenance.get_all_clients()
        assert len(all_clients) == initial_count + 2

        labels = [c.label for c in all_clients]
        assert "test-A" in labels
        assert "test-B" in labels

    def test_get_all_records_aggregates_from_all_clients(self, clean_client_registry):
        """get_all_records() returns records from all clients combined."""
        client_a = covenance.Covenance(label="client-A")
        client_b = covenance.Covenance(label="client-B")

        self._add_mock_record(client_a, model="gpt-4o", provider="openai")
        self._add_mock_record(client_b, model="claude-sonnet-4", provider="anthropic")

        all_records = covenance.get_all_records()

        # Should have 2 records (one from each client)
        assert len(all_records) == 2

        models = {r.model for r in all_records}
        assert models == {"gpt-4o", "claude-sonnet-4"}

    def test_get_all_records_sorted_by_start_time(self, clean_client_registry):
        """get_all_records() returns records sorted by started_at timestamp."""
        import time

        client_a = covenance.Covenance(label="client-A")
        client_b = covenance.Covenance(label="client-B")

        # Add records with slight delay to ensure different timestamps
        self._add_mock_record(client_a, model="model-first", provider="openai")
        time.sleep(0.01)
        self._add_mock_record(client_b, model="model-second", provider="openai")

        all_records = covenance.get_all_records()

        assert all_records[0].model == "model-first"
        assert all_records[1].model == "model-second"

    def test_print_usage_all_clients_aggregates_stats(
        self, clean_client_registry, capsys
    ):
        """print_usage(all_clients=True) shows combined stats from all clients."""
        client_a = covenance.Covenance(label="client-A")
        client_b = covenance.Covenance(label="client-B")

        self._add_mock_record(client_a, model="gpt-4o", provider="openai", tokens=100)
        self._add_mock_record(
            client_b, model="claude-sonnet-4", provider="anthropic", tokens=200
        )

        covenance.print_usage(all_clients=True)

        output = capsys.readouterr().out
        assert "all clients" in output.lower()
        assert "Calls: 2" in output
        # Total tokens: (100 + 50) + (200 + 100) = 450
        assert "450" in output

    def test_print_call_timeline_all_clients(self, clean_client_registry, capsys):
        """print_call_timeline(all_clients=True) shows calls from all clients."""
        client_a = covenance.Covenance(label="client-A")
        client_b = covenance.Covenance(label="client-B")

        self._add_mock_record(client_a, model="gpt-4o", provider="openai")
        self._add_mock_record(client_b, model="gemini-2.5-flash", provider="gemini")

        covenance.print_call_timeline(all_clients=True)

        output = capsys.readouterr().out
        assert "2 calls" in output
        assert "gpt-4o" in output
        assert "g2.5-flash" in output  # Model name gets shortened

    def test_usage_summary_with_all_records(self, clean_client_registry):
        """usage_summary() works correctly with aggregated records."""
        client_a = covenance.Covenance(label="client-A")
        client_b = covenance.Covenance(label="client-B")

        self._add_mock_record(client_a, model="gpt-4o", provider="openai", tokens=100)
        self._add_mock_record(
            client_b, model="claude-sonnet-4", provider="anthropic", tokens=200
        )

        all_records = covenance.get_all_records()
        summary = covenance.usage_summary(records=all_records)

        assert summary["calls"] == 2
        assert summary["tokens_input"] == 300  # 100 + 200
        assert summary["tokens_output"] == 150  # 50 + 100
        assert len(summary["models"]) == 2
