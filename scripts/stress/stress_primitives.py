"""Stress test: non-BaseModel return types.

Tests the ResponseTypeAdapter wrapping/unwrapping for primitive types,
lists, and tuples — common user patterns that bypass Pydantic models.
"""

from stress_utils import DEFAULT_MODEL, StressTestResult, make_client, safe_call


def run_stress_test(
    model: str = DEFAULT_MODEL, backend: str | None = None
) -> StressTestResult:
    """Test primitive and wrapped return types."""
    client = make_client(model, backend)
    failures = []
    details = {}

    tests = [
        (
            "return_int",
            int,
            "How many continents are there? Return just the number.",
            lambda r: (r == 7, f"Expected 7, got {r}"),
        ),
        (
            "return_float",
            float,
            "What is pi rounded to 2 decimal places? Return just the number.",
            lambda r: (abs(r - 3.14) < 0.01, f"Expected ~3.14, got {r}"),
        ),
        (
            "return_bool",
            bool,
            "Is the Earth round? Return true or false.",
            lambda r: (r is True, f"Expected True, got {r}"),
        ),
        (
            "return_list_str",
            list[str],
            "List 3 primary colors.",
            lambda r: (
                isinstance(r, list)
                and len(r) == 3
                and all(isinstance(x, str) for x in r),
                f"Expected list of 3 strings, got {r}",
            ),
        ),
        (
            "return_list_int",
            list[int],
            "Return the first 5 prime numbers as a list.",
            lambda r: (
                r == [2, 3, 5, 7, 11],
                f"Expected [2,3,5,7,11], got {r}",
            ),
        ),
        (
            "return_tuple",
            tuple[str, int, float],
            "Return a tuple of: the word 'hello', the number 42, and pi to 2 decimals.",
            lambda r: (
                isinstance(r, tuple)
                and len(r) == 3
                and r[0].lower() == "hello"
                and r[1] == 42
                and abs(r[2] - 3.14) < 0.01,
                f"Expected ('hello', 42, 3.14), got {r}",
            ),
        ),
    ]

    for name, response_type, prompt, validator in tests:
        result, error = safe_call(
            client,
            lambda rt=response_type, p=prompt: client.ask_llm(
                p, model=model, response_type=rt
            ),
        )
        if error:
            failures.append(f"{name}: {error}")
            details[name] = {"error": error}
        else:
            passed, reason = validator(result)
            if not passed:
                failures.append(f"{name}: {reason}")
                details[name] = {"passed": False, "reason": reason}
            else:
                details[name] = {"passed": True}

    summary = client.usage_summary()
    return StressTestResult(
        name="primitives",
        passed=len(failures) == 0,
        failures=failures,
        calls_made=summary["calls"],
        cost_usd=summary["cost_usd"],
        details=details,
    )


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
