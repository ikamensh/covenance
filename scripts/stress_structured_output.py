"""Stress test for structured LLM output across providers.

Generates diverse Pydantic models programmatically and tests each provider's
ability to return valid structured responses. Uses threading for speed.

Run: python scripts/stress_structured_output.py

Findings (Jan 2026):
- Gemini & Mistral fail with Literal[int] (integer enum values in JSON schema)
- OpenAI handles all tested types correctly
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, create_model

# ============================================================================
# Configuration
# ============================================================================

PROVIDERS = {
    "openai": "gpt-5-nano",
    "gemini": "gemini-2.5-flash-lite",
    "mistral": "mistral-small-latest",
    "anthropic": "claude-haiku-4.5",
}

THREADS_PER_PROVIDER = 3
MODELS_PER_PROVIDER = 20  # How many different schemas to test per provider

# Skip Literal[int] types - Gemini and Mistral can't handle them in JSON schema
SKIP_INT_LITERALS = (
    True  # Set False to test int literals (known to fail on Gemini/Mistral)
)


# ============================================================================
# Model Generation
# ============================================================================

# Building blocks for field types
SIMPLE_TYPES = [str, int, bool, float]
LITERAL_STR_VALUES = [
    Literal["red", "green", "blue"],
    Literal["small", "medium", "large"],
    Literal["pending", "active", "done"],
]
LITERAL_INT_VALUES = [
    Literal[1, 2, 3],  # Known to fail on Gemini/Mistral
]

# Field name pools (semantically meaningful for LLM to fill)
FIELD_POOLS = {
    str: ["name", "title", "description", "category", "label", "color_name"],
    int: ["count", "quantity", "age", "score", "level", "amount"],
    float: ["price", "weight", "rating_score", "temperature", "percentage"],
    bool: ["is_active", "is_valid", "enabled", "completed", "approved"],
    "literal_str": ["status", "priority", "size", "mode", "state"],
    "literal_int": ["rating", "tier", "rank"],
    "list_str": ["tags", "keywords", "labels"],
    "list_int": ["scores", "values", "numbers"],
}


def get_all_types() -> list:
    """Get all type options respecting SKIP_INT_LITERALS setting."""
    types = list(SIMPLE_TYPES) + list(LITERAL_STR_VALUES)

    # List types
    types.extend([list[str], list[int]])

    if not SKIP_INT_LITERALS:
        types.extend(LITERAL_INT_VALUES)

    return types


def generate_model_specs(n: int) -> list[dict]:
    """Generate n diverse model specifications.

    Each spec is a dict of {field_name: field_type} that can be used
    to create a Pydantic model.
    """
    all_types = get_all_types()
    specs = []

    # Strategy: systematically vary number of fields and type combinations
    field_counts = [1, 2, 3, 4, 5]

    idx = 0
    while len(specs) < n:
        num_fields = field_counts[idx % len(field_counts)]
        spec = {}

        for field_idx in range(num_fields):
            # Cycle through type options
            type_choice = (idx + field_idx) % len(all_types)
            field_type = all_types[type_choice]

            # Determine pool key for field name
            origin = get_origin(field_type)
            if origin is Literal:
                literal_args = get_args(field_type)
                pool_key = (
                    "literal_int" if isinstance(literal_args[0], int) else "literal_str"
                )
            elif origin is list:
                elem_type = get_args(field_type)[0]
                pool_key = f"list_{elem_type.__name__}"
            else:
                pool_key = field_type

            # Pick field name from pool
            pool = FIELD_POOLS.get(pool_key, FIELD_POOLS[str])
            field_name = pool[field_idx % len(pool)]

            # Avoid duplicate field names
            base_name = field_name
            suffix = 0
            while field_name in spec:
                suffix += 1
                field_name = f"{base_name}_{suffix}"

            spec[field_name] = field_type

        specs.append(spec)
        idx += 1

    return specs


def create_test_model(spec: dict, model_name: str) -> type[BaseModel]:
    """Create a Pydantic model from a field spec."""
    field_definitions = {name: (typ, ...) for name, typ in spec.items()}
    return create_model(model_name, **field_definitions)


def spec_to_prompt(spec: dict) -> str:
    """Generate a prompt that asks the LLM to fill in the model fields."""
    parts = []
    for name, typ in spec.items():
        origin = get_origin(typ)
        if origin is Literal:
            options = get_args(typ)
            parts.append(f"- {name}: one of {options}")
        elif origin is list:
            elem_type = get_args(typ)[0]
            parts.append(f"- {name}: a list of {elem_type.__name__} values (2-4 items)")
        else:
            parts.append(f"- {name}: a {typ.__name__} value")

    return "Fill in these fields with plausible example values:\n" + "\n".join(parts)


def validate_response(response: BaseModel, spec: dict) -> list[str]:
    """Validate that response matches spec, return list of issues."""
    issues = []
    for name, expected_type in spec.items():
        value = getattr(response, name, None)

        origin = get_origin(expected_type)
        if origin is Literal:
            allowed = get_args(expected_type)
            if value not in allowed:
                issues.append(f"{name}: got {value!r}, expected one of {allowed}")
        elif origin is list:
            if not isinstance(value, list):
                issues.append(f"{name}: got {type(value).__name__}, expected list")
            else:
                elem_type = get_args(expected_type)[0]
                for i, item in enumerate(value):
                    if not isinstance(item, elem_type):
                        issues.append(
                            f"{name}[{i}]: got {type(item).__name__}, expected {elem_type.__name__}"
                        )
                        break  # Report only first bad element
        elif expected_type is float:
            # Accept int as float (common LLM behavior)
            if not isinstance(value, (int, float)):
                issues.append(f"{name}: got {type(value).__name__}, expected float")
        elif not isinstance(value, expected_type):
            issues.append(
                f"{name}: got {type(value).__name__}, expected {expected_type.__name__}"
            )

    return issues


# ============================================================================
# Test Execution
# ============================================================================


@dataclass
class TestResult:
    provider: str
    model_name: str
    spec: dict
    success: bool
    error: str | None = None
    duration_ms: float = 0
    prompt: str = ""
    response_repr: str | None = None  # String repr of response for debugging


def run_single_test(provider: str, model: str, spec: dict, test_id: int) -> TestResult:
    """Run a single structured output test."""
    from covenance import ask_llm

    model_class = create_test_model(spec, f"TestModel_{provider}_{test_id}")
    prompt = spec_to_prompt(spec)

    start = time.perf_counter()
    try:
        response = ask_llm(prompt, model=model, response_type=model_class)
        duration_ms = (time.perf_counter() - start) * 1000
        response_repr = repr(response)

        # Validate the response
        issues = validate_response(response, spec)
        if issues:
            return TestResult(
                provider,
                model,
                spec,
                False,
                f"Validation: {'; '.join(issues)}",
                duration_ms,
                prompt,
                response_repr,
            )

        return TestResult(
            provider, model, spec, True, None, duration_ms, prompt, response_repr
        )

    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        import traceback

        error_detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return TestResult(
            provider, model, spec, False, error_detail, duration_ms, prompt, None
        )


def run_provider_tests(
    provider: str, model: str, specs: list[dict]
) -> list[TestResult]:
    """Run all tests for a single provider using thread pool."""
    results = []

    with ThreadPoolExecutor(max_workers=THREADS_PER_PROVIDER) as executor:
        futures = {
            executor.submit(run_single_test, provider, model, spec, i): i
            for i, spec in enumerate(specs)
        }

        for future in as_completed(futures):
            results.append(future.result())

    return results


def format_type(t) -> str:
    """Format a type for display."""
    origin = get_origin(t)
    if origin is Literal:
        return f"Literal{list(get_args(t))}"
    elif origin is list:
        return f"list[{get_args(t)[0].__name__}]"
    return t.__name__


def print_failure_detail(r: TestResult, verbose: bool = False) -> None:
    """Print detailed failure information for debugging."""
    spec_str = ", ".join(f"{k}: {format_type(v)}" for k, v in r.spec.items())
    print(f"\n    FAIL [{spec_str}]")
    print(f"    Duration: {r.duration_ms:.0f}ms")

    if verbose:
        print(f"    Prompt:\n      {r.prompt.replace(chr(10), chr(10) + '      ')}")
        if r.response_repr:
            resp_str = (
                r.response_repr
                if len(r.response_repr) < 300
                else r.response_repr[:300] + "..."
            )
            print(f"    Response: {resp_str}")

    # Print error, potentially multi-line
    if r.error:
        error_lines = r.error.strip().split("\n")
        # For verbose, show full traceback; otherwise just first few lines
        if verbose:
            for line in error_lines:
                print(f"    {line}")
        else:
            for line in error_lines[:3]:
                print(f"    {line}")
            if len(error_lines) > 3:
                print(f"    ... ({len(error_lines) - 3} more lines)")


def main():
    from covenance import clear_records, print_usage

    all_types = get_all_types()

    print(
        f"Stress testing structured output: {MODELS_PER_PROVIDER} schemas × {len(PROVIDERS)} providers"
    )
    print(f"Threads per provider: {THREADS_PER_PROVIDER}")
    print(f"Types tested: {', '.join(format_type(t) for t in all_types)}")
    if SKIP_INT_LITERALS:
        print("(Skipping Literal[int] - known to fail on Gemini/Mistral)")
    print("-" * 70)

    # Clear any previous records
    clear_records()

    # Generate model specs (same specs for all providers for fair comparison)
    specs = generate_model_specs(MODELS_PER_PROVIDER)

    # Run all providers in parallel
    all_results: dict[str, list[TestResult]] = {}

    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as executor:
        futures = {
            executor.submit(run_provider_tests, provider, model, specs): provider
            for provider, model in PROVIDERS.items()
        }

        for future in as_completed(futures):
            provider = futures[future]
            all_results[provider] = future.result()

            # Print progress
            results = all_results[provider]
            passed = sum(1 for r in results if r.success)
            print(f"  {provider}: {passed}/{len(results)} passed")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    total_passed = 0
    total_tests = 0
    all_failures: list[TestResult] = []

    for provider, results in sorted(all_results.items()):
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        avg_ms = sum(r.duration_ms for r in results) / len(results)
        total_passed += passed
        total_tests += len(results)

        status = "✓" if failed == 0 else "✗"
        print(
            f"\n{status} {provider} ({PROVIDERS[provider]}): {passed}/{len(results)} ({avg_ms:.0f}ms avg)"
        )

        # Collect failures
        failures = [r for r in results if not r.success]
        all_failures.extend(failures)

        # Show brief failure summary per provider
        if failures:
            print(f"    {len(failures)} failures - see details below")

    print(f"\nTotal: {total_passed}/{total_tests} passed")

    # Detailed failure report
    if all_failures:
        print("\n" + "=" * 70)
        print(f"FAILURE DETAILS ({len(all_failures)} failures)")
        print("=" * 70)

        for r in all_failures[:10]:  # Show up to 10 detailed failures
            print(f"\n[{r.provider}]")
            print_failure_detail(r, verbose=True)

        if len(all_failures) > 10:
            print(f"\n... and {len(all_failures) - 10} more failures (not shown)")

    # Usage summary
    print_usage(title="LLM Usage Summary")


if __name__ == "__main__":
    main()
