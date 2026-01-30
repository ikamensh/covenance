"""Property tests for retry utilities."""

from unittest.mock import patch

from covenance.retry import exponential_backoff


def test_exponential_backoff_properties():
    """Property test: exponential backoff increases exponentially and respects bounds."""
    # Test exponential growth
    wait_times = [exponential_backoff(i) for i in range(5)]

    # Each wait time should be greater than the previous (on average, accounting for jitter)
    # Since jitter adds randomness, we test bounds instead
    for i in range(4):
        # Minimum possible wait (50% jitter): base * 2^i * 0.5
        min_wait = 1.0 * (2**i) * 0.5
        # Maximum possible wait (100% jitter): base * 2^i * 1.0
        max_wait = 1.0 * (2**i) * 1.0

        assert min_wait <= wait_times[i] <= max_wait, (
            f"Wait time {wait_times[i]} for attempt {i} should be between "
            f"{min_wait} and {max_wait}"
        )


def test_exponential_backoff_respects_max_wait():
    """Property test: exponential backoff caps at max_wait."""
    # Test with small max_wait to ensure capping works
    max_wait = 5.0
    wait_times = [exponential_backoff(i, max_wait=max_wait) for i in range(10)]

    for wait_time in wait_times:
        # Maximum possible wait with jitter: max_wait * 1.0
        assert wait_time <= max_wait, (
            f"Wait time {wait_time} should not exceed max_wait {max_wait}"
        )


def test_exponential_backoff_minimum_wait():
    """Property test: exponential backoff never returns less than 0.1 seconds."""
    wait_times = [exponential_backoff(i, base_wait=0.01) for i in range(5)]

    for wait_time in wait_times:
        assert wait_time >= 0.1, f"Wait time {wait_time} should be at least 0.1 seconds"


def test_exponential_backoff_jitter_randomization():
    """Property test: jitter adds randomization between 50% and 100%."""
    # Use fixed seed to test jitter range
    with patch("random.random", return_value=0.5):
        wait_time = exponential_backoff(2, base_wait=1.0)
        # With attempt=2: 1.0 * 2^2 = 4.0
        # With jitter factor 0.5 + (0.5 * 0.5) = 0.75: 4.0 * 0.75 = 3.0
        expected = 4.0 * 0.75
        assert wait_time == expected

    with patch("random.random", return_value=1.0):
        wait_time = exponential_backoff(2, base_wait=1.0)
        # With jitter factor 0.5 + (1.0 * 0.5) = 1.0: 4.0 * 1.0 = 4.0
        expected = 4.0 * 1.0
        assert wait_time == expected

    with patch("random.random", return_value=0.0):
        wait_time = exponential_backoff(2, base_wait=1.0)
        # With jitter factor 0.5 + (0.0 * 0.5) = 0.5: 4.0 * 0.5 = 2.0
        expected = 4.0 * 0.5
        assert wait_time == expected


def test_exponential_backoff_custom_base_wait():
    """Property test: exponential backoff respects custom base_wait."""
    base_wait = 2.0
    wait_times = [exponential_backoff(i, base_wait=base_wait) for i in range(3)]

    for i, wait_time in enumerate(wait_times):
        # Minimum: base_wait * 2^i * 0.5
        min_wait = base_wait * (2**i) * 0.5
        assert wait_time >= min_wait, (
            f"Wait time {wait_time} for attempt {i} should be at least {min_wait}"
        )
