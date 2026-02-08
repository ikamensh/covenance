"""Run all structured output stress tests across providers."""

import importlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from covenance import print_call_timeline, print_usage

# Add script directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

# Models to test: cheapest and next-to-cheapest per provider
MODELS_TO_TEST = [
    # OpenAI
    ("openai", "gpt-4.1-nano"),
    ("openai", "gpt-4.1-mini"),
    # Google
    ("google", "gemini-2.5-flash-lite"),
    ("google", "gemini-2.0-flash"),
    # Mistral (probabilistic, interesting to compare)
    ("mistral", "mistral-small-latest"),
]

# Which tests to run
TEST_MODULES = [
    "stress_nesting_depth",
    "stress_field_width",
    "stress_recursive",
    "stress_type_diversity",
    "stress_enums",
    "stress_optionals",
    "stress_field_constraints",
    "stress_cross_field",
    "stress_edge_values",
    "stress_consistency",
    "stress_real_schemas",
    "stress_limits",  # Push to find breaking points
]


@dataclass
class Progress:
    """Thread-safe progress tracker."""

    total: int
    completed: int = 0
    lock: threading.Lock = None

    def __post_init__(self):
        self.lock = threading.Lock()

    def increment(self) -> int:
        with self.lock:
            self.completed += 1
            return self.completed

    def status(self) -> str:
        with self.lock:
            pct = (self.completed / self.total) * 100 if self.total > 0 else 0
            return f"[{self.completed}/{self.total}] {pct:.0f}%"


def run_single_test(
    test_name: str, model: str, progress: Progress
) -> tuple[str, str, any]:
    """Run a single test and return (test_name, model, result)."""
    try:
        module = importlib.import_module(test_name)
        result = module.run_stress_test(model=model)
        progress.increment()

        status = "PASS" if result.passed else "FAIL"
        short_model = model.split("-")[-1][:6]
        print(
            f"  {progress.status()}  {status}  {result.name:<18}  {short_model:<8}  ${result.cost_usd:.4f}"
        )

        return test_name, model, result
    except Exception as e:
        progress.increment()
        print(
            f"  {progress.status()}  ERR   {test_name:<18}  {model:<8}  {type(e).__name__}"
        )
        return test_name, model, {"error": str(e)}


def run_all_tests(
    models: list[tuple[str, str]] | None = None,
    tests: list[str] | None = None,
    max_workers: int = 12,
):
    """Run stress tests in parallel across models and tests.

    Args:
        models: List of (provider, model) tuples. None = all default models.
        tests: List of test module names. None = all tests.
        max_workers: Max parallel workers. Default 8 balances speed vs rate limits.
    """
    if models is None:
        models = MODELS_TO_TEST
    if tests is None:
        tests = TEST_MODULES

    # Build work items: all (test, model) combinations
    work_items = [(test, model) for _, model in models for test in tests]
    total_tasks = len(work_items)

    print(f"\nRunning {total_tasks} tests ({len(tests)} tests × {len(models)} models)")
    print(f"Parallelism: {max_workers} workers")
    print("-" * 60)

    progress = Progress(total=total_tasks)
    start_time = time.time()

    # Results: {model: {test_name: result}}
    all_results: dict[str, dict[str, any]] = {model: {} for _, model in models}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_test, test, model, progress): (test, model)
            for test, model in work_items
        }

        for future in as_completed(futures):
            test_name, model, result = future.result()

            # Store result by the result's name if available, otherwise test module name
            if hasattr(result, "name"):
                all_results[model][result.name] = result
            else:
                all_results[model][test_name] = result

    elapsed = time.time() - start_time

    # Calculate totals
    total_cost = 0.0
    total_calls = 0
    for model_results in all_results.values():
        for r in model_results.values():
            if hasattr(r, "cost_usd"):
                total_cost += r.cost_usd
                total_calls += r.calls_made

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")

    # Build summary table
    print(f"\n{'Test':<22}", end="")
    for _, model in models:
        short = model.replace("mistral-small-latest", "mistral").replace("-latest", "")
        short = short[:10]
        print(f"{short:>12}", end="")
    print()
    print("-" * (22 + 12 * len(models)))

    # Build mapping from module name to result name
    # e.g., "stress_enums" -> "enum_adherence", "stress_optionals" -> "optional_fields"
    module_to_result = {
        "stress_nesting_depth": "nesting depth",
        "stress_field_width": "field width",
        "stress_recursive": "recursive types",
        "stress_type_diversity": "type diversity",
        "stress_enums": "enum adherence",
        "stress_optionals": "optional fields",
        "stress_field_constraints": "field constraints",
        "stress_cross_field": "cross field deps",
        "stress_edge_values": "edge values",
        "stress_consistency": "consistency",
        "stress_real_schemas": "real schemas",
        "stress_limits": "limits",
    }

    for test_module in tests:
        display_name = test_module.replace("stress_", "")[:20]
        print(f"{display_name:<22}", end="")

        for _, model in models:
            model_results = all_results.get(model, {})

            # Find matching result using the mapping
            result = None
            expected_name = module_to_result.get(test_module, "")

            for name, r in model_results.items():
                if hasattr(r, "name"):
                    # Normalize both for comparison
                    result_norm = r.name.replace("_", " ")
                    if result_norm == expected_name:
                        result = r
                        break

            if result is not None and hasattr(result, "passed"):
                symbol = "PASS" if result.passed else "FAIL"
            elif isinstance(result, dict) and "error" in result:
                symbol = "ERR"
            else:
                symbol = "-"

            print(f"{symbol:>12}", end="")
        print()

    print("-" * (22 + 12 * len(models)))

    # Totals
    print(f"\nTotal calls: {total_calls}")
    print(f"Total cost:  ${total_cost:.4f}")
    print(f"Elapsed:     {elapsed:.1f}s")

    # Pass rates per model
    print("\nPass rates:")
    for _, model in models:
        results = all_results.get(model, {})
        passed = sum(1 for r in results.values() if hasattr(r, "passed") and r.passed)
        total = sum(1 for r in results.values() if hasattr(r, "passed"))
        short = model.replace("mistral-small-latest", "mistral-small")[:20]
        print(f"  {short:<22} {passed}/{total}")

    # Overall
    all_passed = all(
        r.passed
        for model_results in all_results.values()
        for r in model_results.values()
        if hasattr(r, "passed")
    )

    return all_results, all_passed


def run_quick_test(model: str = "gpt-4.1-nano"):
    """Run a quick subset of tests on one model."""
    quick_tests = [
        "stress_nesting_depth",
        "stress_type_diversity",
        "stress_field_constraints",
    ]

    provider = "openai"
    if model.startswith("gemini"):
        provider = "google"
    elif model.startswith("mistral"):
        provider = "mistral"
    elif model.startswith("claude"):
        provider = "anthropic"

    return run_all_tests(models=[(provider, model)], tests=quick_tests, max_workers=3)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4.1-nano"
        results, passed = run_quick_test(model)
    else:
        results, passed = run_all_tests()
    # print_usage(all_clients=True)
    # print_call_timeline(all_clients=True)

    sys.exit(0 if passed else 1)
