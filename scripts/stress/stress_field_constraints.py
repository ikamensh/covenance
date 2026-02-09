"""Stress test: field description constraints."""

from pydantic import BaseModel, Field
from stress_utils import DEFAULT_MODEL, StressTestResult, make_client, run_test_cases


class BoundedNumbers(BaseModel):
    """Numbers with range constraints in descriptions."""

    temperature: int = Field(
        description="Temperature in Celsius, must be between -40 and 50"
    )
    percentage: float = Field(
        description="A percentage value, must be between 0.0 and 100.0"
    )
    dice_roll: int = Field(description="Result of rolling a 6-sided die, must be 1-6")
    age: int = Field(description="Human age in years, must be between 0 and 120")


class SpecificValues(BaseModel):
    """Fields that must have specific values."""

    magic_number: int = Field(description="Must be exactly 42")
    greeting: str = Field(description="Must be exactly 'Hello, World!'")
    pi_approx: float = Field(description="Must be 3.14159 (five decimal places)")


class LogicalRules(BaseModel):
    """Fields with logical constraints."""

    numerator: int = Field(description="A positive integer for division")
    denominator: int = Field(description="A positive integer, must NOT be zero")
    even_number: int = Field(description="Must be an even number between 2 and 100")
    odd_number: int = Field(description="Must be an odd number between 1 and 99")


def validate_bounded(result: BoundedNumbers) -> tuple[bool, str]:
    """Validate all bounds are respected."""
    issues = []

    if not (-40 <= result.temperature <= 50):
        issues.append(f"temperature {result.temperature} not in [-40, 50]")
    if not (0.0 <= result.percentage <= 100.0):
        issues.append(f"percentage {result.percentage} not in [0, 100]")
    if not (1 <= result.dice_roll <= 6):
        issues.append(f"dice_roll {result.dice_roll} not in [1, 6]")
    if not (0 <= result.age <= 120):
        issues.append(f"age {result.age} not in [0, 120]")

    if issues:
        return False, f"Bound violations: {issues}"
    return True, ""


def validate_specific(result: SpecificValues) -> tuple[bool, str]:
    """Validate specific values are exact."""
    issues = []

    if result.magic_number != 42:
        issues.append(f"magic_number is {result.magic_number}, expected 42")
    if result.greeting != "Hello, World!":
        issues.append(f"greeting is '{result.greeting}', expected 'Hello, World!'")
    if abs(result.pi_approx - 3.14159) > 0.000001:
        issues.append(f"pi_approx is {result.pi_approx}, expected 3.14159")

    if issues:
        return False, f"Value mismatches: {issues}"
    return True, ""


def validate_logical(result: LogicalRules) -> tuple[bool, str]:
    """Validate logical constraints."""
    issues = []

    if result.numerator <= 0:
        issues.append(f"numerator {result.numerator} not positive")
    if result.denominator <= 0:
        issues.append(f"denominator {result.denominator} not positive")
    if result.denominator == 0:
        issues.append("denominator is zero")
    if result.even_number % 2 != 0 or not (2 <= result.even_number <= 100):
        issues.append(f"even_number {result.even_number} invalid")
    if result.odd_number % 2 != 1 or not (1 <= result.odd_number <= 99):
        issues.append(f"odd_number {result.odd_number} invalid")

    if issues:
        return False, f"Logical violations: {issues}"
    return True, ""


def run_stress_test(
    model: str = DEFAULT_MODEL, backend: str | None = None
) -> StressTestResult:
    """Test field description constraint adherence."""
    client = make_client(model, backend)

    cases = [
        (
            "bounded_numbers",
            lambda: client.ask_llm(
                "Generate values that satisfy all the field constraints described in the schema.",
                model=model,
                response_type=BoundedNumbers,
            ),
            validate_bounded,
        ),
        (
            "specific_values",
            lambda: client.ask_llm(
                "Fill in the exact values specified in each field's description.",
                model=model,
                response_type=SpecificValues,
            ),
            validate_specific,
        ),
        (
            "logical_rules",
            lambda: client.ask_llm(
                "Generate numbers that satisfy all the logical constraints in the field descriptions.",
                model=model,
                response_type=LogicalRules,
            ),
            validate_logical,
        ),
    ]

    return run_test_cases(client, "field_constraints", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
