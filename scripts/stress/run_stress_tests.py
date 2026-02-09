"""Run structured output stress tests across providers and backends.

Usage:
    python run_stress_tests.py                    # default: auto backend
    python run_stress_tests.py --backend both     # compare native vs pydantic
    python run_stress_tests.py --backend native   # native only
    python run_stress_tests.py --backend pydantic # pydantic-ai only
    python run_stress_tests.py --quick            # quick subset, one model
    python run_stress_tests.py --quick gpt-4.1-nano
"""

import importlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Models to test: cheapest per provider
MODELS_TO_TEST = [
    ("openai", "gpt-4.1-nano"),
    ("google", "gemini-2.5-flash-lite"),
    ("mistral", "mistral-small-latest"),
    ("grok", "grok-3-mini"),
    ("anthropic", "claude-haiku-4-5"),
]

# All test modules
TEST_MODULES = [
    "stress_primitives",
    "stress_type_diversity",
    "stress_unions",
    "stress_enums",
    "stress_optionals",
    "stress_nesting_depth",
    "stress_field_width",
    "stress_recursive",
    "stress_field_constraints",
    "stress_cross_field",
    "stress_edge_values",
    "stress_formats",
    "stress_real_schemas",
    "stress_limits",
]

# Maps module name -> result.name (what run_test_cases returns)
MODULE_RESULT_NAME = {
    "stress_primitives": "primitives",
    "stress_type_diversity": "type_diversity",
    "stress_unions": "unions",
    "stress_enums": "enum_adherence",
    "stress_optionals": "optional_fields",
    "stress_nesting_depth": "nesting_depth",
    "stress_field_width": "field_width",
    "stress_recursive": "recursive_types",
    "stress_field_constraints": "field_constraints",
    "stress_cross_field": "cross_field_deps",
    "stress_edge_values": "edge_values",
    "stress_formats": "formats",
    "stress_real_schemas": "real_schemas",
    "stress_limits": "limits",
}


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
    test_name: str, model: str, backend: str | None, progress: Progress
) -> tuple[str, str, str | None, any]:
    """Run a single test. Returns (test_name, model, backend, result)."""
    try:
        module = importlib.import_module(test_name)
        result = module.run_stress_test(model=model, backend=backend)
        progress.increment()

        status = "PASS" if result.passed else "FAIL"
        backend_tag = f"/{backend}" if backend else ""
        short_model = model.split("-")[-1][:8]
        print(
            f"  {progress.status()}  {status}  {result.name:<16}  "
            f"{short_model}{backend_tag:<16}  ${result.cost_usd:.4f}"
        )

        return test_name, model, backend, result
    except Exception as e:
        progress.increment()
        backend_tag = f"/{backend}" if backend else ""
        print(
            f"  {progress.status()}  ERR   {test_name:<16}  "
            f"{model}{backend_tag:<16}  {type(e).__name__}: {str(e)[:60]}"
        )
        return test_name, model, backend, {"error": str(e)}


def run_all_tests(
    models: list[tuple[str, str]] | None = None,
    tests: list[str] | None = None,
    backends: list[str | None] | None = None,
    max_workers: int = 10,
):
    """Run stress tests across models, tests, and optionally backends."""
    if models is None:
        models = MODELS_TO_TEST
    if tests is None:
        tests = TEST_MODULES
    if backends is None:
        backends = [None]  # auto routing

    work_items = [
        (test, model, backend)
        for _, model in models
        for test in tests
        for backend in backends
    ]
    total_tasks = len(work_items)

    backend_desc = "/".join(b or "auto" for b in backends)
    print(
        f"\nRunning {total_tasks} tests ({len(tests)} tests x {len(models)} models x {backend_desc})"
    )
    print(f"Parallelism: {max_workers} workers")
    print("-" * 70)

    progress = Progress(total=total_tasks)
    start_time = time.time()

    # Results: {(model, backend): {test_name: result}}
    all_results: dict[tuple[str, str | None], dict[str, any]] = {
        (model, backend): {} for _, model in models for backend in backends
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_test, test, model, backend, progress): (
                test,
                model,
                backend,
            )
            for test, model, backend in work_items
        }

        for future in as_completed(futures):
            test_name, model, backend, result = future.result()
            key = (model, backend)
            result_name = result.name if hasattr(result, "name") else test_name
            all_results[key][result_name] = result

    elapsed = time.time() - start_time

    # Calculate totals
    total_cost = 0.0
    total_calls = 0
    for model_results in all_results.values():
        for r in model_results.values():
            if hasattr(r, "cost_usd"):
                total_cost += r.cost_usd
                total_calls += r.calls_made

    # Print summary table
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")

    # Column headers
    columns = [(model, backend) for _, model in models for backend in backends]
    col_width = 14

    def col_label(model, backend):
        short = model.replace("mistral-small-latest", "mistral-sm")
        short = short.replace("gemini-2.5-flash-lite", "gem-flash")
        short = short.replace("gpt-4.1-nano", "gpt-nano")
        if backend:
            short = f"{short[:6]}/{backend[:3]}"
        return short[:col_width]

    print(f"\n{'Test':<16}", end="")
    for model, backend in columns:
        print(f"{col_label(model, backend):>{col_width}}", end="")
    print()
    print("-" * (16 + col_width * len(columns)))

    for test_module in tests:
        display = test_module.replace("stress_", "")[:15]
        print(f"{display:<16}", end="")

        for model, backend in columns:
            results = all_results.get((model, backend), {})

            # Find the matching result by result.name
            result = None
            expected = MODULE_RESULT_NAME.get(test_module, "")
            for name, r in results.items():
                if hasattr(r, "name") and r.name == expected:
                    result = r
                    break

            if result is not None and hasattr(result, "passed"):
                symbol = "PASS" if result.passed else "FAIL"
                # Show failure count for partial failures
                if not result.passed and result.failures:
                    n_fail = len(result.failures)
                    n_total = len(result.details) if result.details else n_fail
                    symbol = f"{n_total - n_fail}/{n_total}"
            elif isinstance(result, dict) and "error" in result:
                symbol = "ERR"
            else:
                symbol = "-"

            print(f"{symbol:>{col_width}}", end="")
        print()

    print("-" * (16 + col_width * len(columns)))

    # Pass rates
    print(f"\nTotal calls: {total_calls}")
    print(f"Total cost:  ${total_cost:.4f}")
    print(f"Elapsed:     {elapsed:.1f}s")

    print("\nPass rates:")
    for model, backend in columns:
        results = all_results.get((model, backend), {})
        passed = sum(1 for r in results.values() if hasattr(r, "passed") and r.passed)
        total = sum(1 for r in results.values() if hasattr(r, "passed"))
        label = col_label(model, backend)
        print(f"  {label:<20} {passed}/{total}")

    # If comparing backends, show diffs
    if len(backends) > 1:
        print("\n  Backend differences:")
        for _, model in models:
            for test_module in tests:
                expected = MODULE_RESULT_NAME.get(test_module, "")
                results_by_backend = {}
                for backend in backends:
                    results = all_results.get((model, backend), {})
                    for name, r in results.items():
                        if hasattr(r, "name") and r.name == expected:
                            results_by_backend[backend or "auto"] = r
                            break

                if len(results_by_backend) >= 2:
                    statuses = {
                        b: r.passed
                        for b, r in results_by_backend.items()
                        if hasattr(r, "passed")
                    }
                    if len(set(statuses.values())) > 1:
                        short_model = model.split("-")[-1][:8]
                        test_display = test_module.replace("stress_", "")
                        diff_parts = [
                            f"{b}={'PASS' if p else 'FAIL'}"
                            for b, p in statuses.items()
                        ]
                        print(
                            f"    {short_model}/{test_display}: {', '.join(diff_parts)}"
                        )

    all_passed = all(
        r.passed
        for model_results in all_results.values()
        for r in model_results.values()
        if hasattr(r, "passed")
    )

    return all_results, all_passed


def run_quick_test(model: str = "gpt-4.1-nano", backend: str | None = None):
    """Run a quick subset on one model."""
    quick_tests = [
        "stress_primitives",
        "stress_unions",
        "stress_type_diversity",
        "stress_formats",
        "stress_field_constraints",
    ]

    provider = "openai"
    if model.startswith("gemini"):
        provider = "google"
    elif model.startswith("mistral"):
        provider = "mistral"
    elif model.startswith("claude"):
        provider = "anthropic"
    elif model.startswith("grok"):
        provider = "grok"

    backends = [backend] if backend != "both" else ["native", "pydantic"]
    return run_all_tests(
        models=[(provider, model)], tests=quick_tests, max_workers=5, backends=backends
    )


def print_failures(all_results):
    """Print detailed failure info."""
    for (model, backend), results in all_results.items():
        for name, r in results.items():
            if hasattr(r, "passed") and not r.passed:
                backend_tag = f"/{backend}" if backend else ""
                print(f"\n  {model}{backend_tag} / {name}:")
                for f in r.failures:
                    print(f"    - {f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--backend", choices=["native", "pydantic", "both"], default=None
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--failures", action="store_true", help="Print failure details")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    if args.quick:
        model = args.model or "gpt-4.1-nano"
        results, passed = run_quick_test(model, backend=args.backend)
    else:
        backends = ["native", "pydantic"] if args.backend == "both" else [args.backend]
        models = None
        if args.model:
            provider = "openai"
            if args.model.startswith("gemini"):
                provider = "google"
            elif args.model.startswith("mistral"):
                provider = "mistral"
            models = [(provider, args.model)]
        results, passed = run_all_tests(
            models=models, backends=backends, max_workers=args.workers
        )

    if args.failures:
        print_failures(results)

    sys.exit(0 if passed else 1)
