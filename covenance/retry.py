"""Shared retry utilities for LLM clients."""

import random


def exponential_backoff(
    attempt: int, base_wait: float = 1.0, max_wait: float = 60.0
) -> float:
    """Calculate exponential backoff wait time with jitter.

    Uses exponential backoff: wait = base_wait * (2 ** attempt)
    Adds jitter to avoid thundering herd problem.
    Caps at max_wait to prevent excessive delays.

    Args:
        attempt: Current attempt number (0-indexed)
        base_wait: Base wait time in seconds (default: 1.0)
        max_wait: Maximum wait time in seconds (default: 60.0)

    Returns:
        Wait time in seconds with jitter applied
    """
    # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s, 60s, ...
    exponential_wait = base_wait * (2**attempt)
    capped_wait = min(exponential_wait, max_wait)

    # Add jitter: randomize between 50% and 100% of calculated wait time
    jitter_factor = 0.5 + (random.random() * 0.5)
    wait_time = capped_wait * jitter_factor

    return max(wait_time, 0.1)
