"""Common utilities for structured output stress tests."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

from covenance import Covenance


@dataclass
class StressTestResult:
    """Result of a single stress test."""

    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    calls_made: int = 0
    cost_usd: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def safe_call(
    client: Covenance, call_fn: Callable, timeout_seconds: int = 60
) -> tuple[Any, str | None]:
    """Execute a call with timeout, returning (result, error_message)."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call_fn)
        try:
            result = future.result(timeout=timeout_seconds)
            return result, None
        except FuturesTimeoutError:
            return None, f"Timeout after {timeout_seconds}s"
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)[:200]}"


def run_test_cases(
    client: Covenance,
    test_name: str,
    cases: list[tuple[str, Callable, Callable[[Any], tuple[bool, str]]]],
    timeout_seconds: int = 30,
) -> StressTestResult:
    """Run multiple test cases and aggregate results."""
    failures = []
    details = {}

    for case_name, call_fn, validator in cases:
        result, error = safe_call(client, call_fn, timeout_seconds)

        if error:
            failures.append(f"{case_name}: {error}")
            details[case_name] = {"error": error}
        else:
            passed, reason = validator(result)
            if not passed:
                failures.append(f"{case_name}: {reason}")
                details[case_name] = {
                    "passed": False,
                    "reason": reason,
                    "result": str(result)[:200],
                }
            else:
                details[case_name] = {"passed": True}

    summary = client.usage_summary()

    return StressTestResult(
        name=test_name,
        passed=len(failures) == 0,
        failures=failures,
        calls_made=summary["calls"],
        cost_usd=summary["cost_usd"],
        details=details,
    )


def make_client(model: str, backend: str | None = None) -> Covenance:
    """Create a Covenance client with optional backend forcing."""
    label_parts = ["stress"]
    if backend:
        label_parts.append(backend)
    client = Covenance(label="_".join(label_parts))
    if backend:
        client.backends.set_all(backend)
    return client


# Cheap models for stress testing
CHEAP_MODELS = {
    "openai": ["gpt-4.1-nano", "gpt-4.1-mini"],
    "google": ["gemini-2.5-flash-lite", "gemini-2.0-flash"],
    "mistral": ["mistral-small-latest"],
}

DEFAULT_MODEL = "gpt-4.1-nano"
