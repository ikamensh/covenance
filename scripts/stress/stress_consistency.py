"""Stress test: output consistency across repeated calls.

Reports numerical consistency rates rather than binary pass/fail.
"""

from pydantic import BaseModel
from stress_utils import DEFAULT_MODEL, StressTestResult, safe_call

from covenance import Covenance


class FactAnswer(BaseModel):
    """Answer to a factual question."""

    answer: str


class MathResult(BaseModel):
    """Result of a math calculation."""

    result: int


class StructuredFact(BaseModel):
    """A structured factual response."""

    capital: str
    country: str


NUM_TRIALS = 5  # Number of times to repeat each test
PASS_THRESHOLD = 0.8  # 80% consistency required to pass


def run_stress_test(model: str = DEFAULT_MODEL) -> StressTestResult:
    """Test consistency of repeated identical calls.

    Reports consistency rates like "4/5 consistent" rather than pass/fail.
    """
    client = Covenance(label="stress_consistency")
    failures = []
    details = {}

    # Test 1: Factual question consistency
    fact_results = []
    for i in range(NUM_TRIALS):
        result, error = safe_call(
            client,
            lambda: client.ask_llm(
                "What is the capital of France? Answer with just the city name.",
                model=model,
                response_type=FactAnswer,
                temperature=0,
            ),
            timeout_seconds=30,
        )
        if error:
            fact_results.append(f"ERROR:{error[:20]}")
        else:
            fact_results.append(result.answer.lower().strip())

    fact_consistency = calculate_consistency(fact_results)
    details["fact_consistency"] = {
        "rate": fact_consistency,
        "values": fact_results,
        "description": f"{int(fact_consistency * NUM_TRIALS)}/{NUM_TRIALS} identical",
    }
    if fact_consistency < PASS_THRESHOLD:
        failures.append(
            f"fact_consistency: {fact_consistency:.0%} < {PASS_THRESHOLD:.0%} threshold"
        )

    # Test 2: Math calculation consistency
    math_results = []
    correct_answer = 17 * 23  # 391
    for i in range(NUM_TRIALS):
        result, error = safe_call(
            client,
            lambda: client.ask_llm(
                "Calculate 17 * 23. Return only the numeric result.",
                model=model,
                response_type=MathResult,
                temperature=0,
            ),
            timeout_seconds=30,
        )
        if error:
            math_results.append(-1)
        else:
            math_results.append(result.result)

    math_consistency = calculate_consistency(math_results)
    math_correctness = sum(1 for r in math_results if r == correct_answer) / len(
        math_results
    )

    details["math_consistency"] = {
        "rate": math_consistency,
        "correctness": math_correctness,
        "values": math_results,
        "expected": correct_answer,
        "description": f"{int(math_consistency * NUM_TRIALS)}/{NUM_TRIALS} identical, {int(math_correctness * NUM_TRIALS)}/{NUM_TRIALS} correct",
    }
    if math_consistency < PASS_THRESHOLD:
        failures.append(
            f"math_consistency: {math_consistency:.0%} < {PASS_THRESHOLD:.0%} threshold"
        )
    if math_correctness < PASS_THRESHOLD:
        failures.append(
            f"math_correctness: {math_correctness:.0%} < {PASS_THRESHOLD:.0%} threshold"
        )

    # Test 3: Structured fact consistency
    struct_results = []
    for i in range(NUM_TRIALS):
        result, error = safe_call(
            client,
            lambda: client.ask_llm(
                "What is the capital of Japan? Give the city and country.",
                model=model,
                response_type=StructuredFact,
                temperature=0,
            ),
            timeout_seconds=30,
        )
        if error:
            struct_results.append(("ERROR", "ERROR"))
        else:
            struct_results.append(
                (result.capital.lower().strip(), result.country.lower().strip())
            )

    struct_consistency = calculate_consistency(struct_results)
    details["struct_consistency"] = {
        "rate": struct_consistency,
        "values": struct_results,
        "description": f"{int(struct_consistency * NUM_TRIALS)}/{NUM_TRIALS} identical",
    }
    if struct_consistency < PASS_THRESHOLD:
        failures.append(
            f"struct_consistency: {struct_consistency:.0%} < {PASS_THRESHOLD:.0%} threshold"
        )

    # Calculate overall consistency
    overall = (fact_consistency + math_consistency + struct_consistency) / 3
    details["overall_consistency"] = f"{overall:.0%}"

    summary = client.usage_summary()

    return StressTestResult(
        name="consistency",
        passed=len(failures) == 0,
        failures=failures,
        calls_made=summary["calls"],
        cost_usd=summary["cost_usd"],
        details=details,
    )


def calculate_consistency(results: list) -> float:
    """Calculate what fraction of results match the most common value."""
    if not results:
        return 0.0
    from collections import Counter

    counts = Counter(str(r) for r in results)  # Stringify for hashability
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(results)


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    result = run_stress_test(model)

    print(f"\n{'PASS' if result.passed else 'FAIL'}: {result.name}")
    print(f"Calls: {result.calls_made}, Cost: ${result.cost_usd:.4f}")
    print("\nConsistency rates:")
    for key, val in result.details.items():
        if isinstance(val, dict) and "description" in val:
            print(f"  {key}: {val['description']}")
        elif key == "overall_consistency":
            print(f"  {key}: {val}")

    if result.failures:
        print("\nFailures:")
        for f in result.failures:
            print(f"  - {f}")
