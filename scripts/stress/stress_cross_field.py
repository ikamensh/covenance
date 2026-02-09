"""Stress test: cross-field dependency constraints."""

from pydantic import BaseModel, Field
from stress_utils import DEFAULT_MODEL, StressTestResult, make_client, run_test_cases


class DateRange(BaseModel):
    """Range where start must precede end."""

    start_year: int = Field(description="Starting year, must be before end_year")
    end_year: int = Field(description="Ending year, must be after start_year")
    start_month: int = Field(description="Starting month (1-12)")
    end_month: int = Field(
        description="Ending month (1-12), in the same or later year-month as start"
    )


class MinMaxPair(BaseModel):
    """Paired values with ordering constraint."""

    min_value: float = Field(description="Minimum value, must be less than max_value")
    max_value: float = Field(
        description="Maximum value, must be greater than min_value"
    )
    current_value: float = Field(
        description="Current value, must be between min_value and max_value"
    )


class ParentChild(BaseModel):
    """Parent-child relationship with age constraints."""

    parent_age: int = Field(description="Parent's age in years")
    child_age: int = Field(
        description="Child's age, must be at least 15 years less than parent's age"
    )
    grandparent_age: int = Field(
        description="Grandparent's age, must be at least 15 years more than parent's age"
    )


class Coordinates(BaseModel):
    """Coordinate system with constraints."""

    origin_x: float = Field(description="Origin X coordinate")
    origin_y: float = Field(description="Origin Y coordinate")
    point_x: float = Field(description="Point X, must be different from origin_x")
    point_y: float = Field(description="Point Y, must be different from origin_y")
    distance: float = Field(
        description="Euclidean distance from origin to point, must be positive"
    )


def validate_date_range(result: DateRange) -> tuple[bool, str]:
    """Validate date ordering."""
    issues = []

    if result.start_year > result.end_year:
        issues.append(f"start_year {result.start_year} > end_year {result.end_year}")
    elif result.start_year == result.end_year and result.start_month > result.end_month:
        issues.append(
            f"same year but start_month {result.start_month} > end_month {result.end_month}"
        )

    if not (1 <= result.start_month <= 12):
        issues.append(f"start_month {result.start_month} not in [1,12]")
    if not (1 <= result.end_month <= 12):
        issues.append(f"end_month {result.end_month} not in [1,12]")

    if issues:
        return False, f"Date range violations: {issues}"
    return True, ""


def validate_minmax(result: MinMaxPair) -> tuple[bool, str]:
    """Validate min/max ordering."""
    issues = []

    if result.min_value >= result.max_value:
        issues.append(f"min {result.min_value} >= max {result.max_value}")
    if not (result.min_value <= result.current_value <= result.max_value):
        issues.append(
            f"current {result.current_value} not in [{result.min_value}, {result.max_value}]"
        )

    if issues:
        return False, f"Min/max violations: {issues}"
    return True, ""


def validate_ages(result: ParentChild) -> tuple[bool, str]:
    """Validate age relationships."""
    issues = []

    parent_child_gap = result.parent_age - result.child_age
    grandparent_parent_gap = result.grandparent_age - result.parent_age

    if parent_child_gap < 15:
        issues.append(f"parent-child gap {parent_child_gap} < 15")
    if grandparent_parent_gap < 15:
        issues.append(f"grandparent-parent gap {grandparent_parent_gap} < 15")
    if result.child_age < 0:
        issues.append(f"child_age {result.child_age} < 0")

    if issues:
        return False, f"Age violations: {issues}"
    return True, ""


def validate_coordinates(result: Coordinates) -> tuple[bool, str]:
    """Validate coordinate constraints."""
    issues = []

    if result.point_x == result.origin_x:
        issues.append("point_x equals origin_x")
    if result.point_y == result.origin_y:
        issues.append("point_y equals origin_y")
    if result.distance <= 0:
        issues.append(f"distance {result.distance} not positive")

    # Check if distance is approximately correct
    import math

    actual_distance = math.sqrt(
        (result.point_x - result.origin_x) ** 2
        + (result.point_y - result.origin_y) ** 2
    )
    if abs(actual_distance - result.distance) > 0.1:
        issues.append(
            f"distance {result.distance} doesn't match actual {actual_distance:.2f}"
        )

    if issues:
        return False, f"Coordinate violations: {issues}"
    return True, ""


def run_stress_test(
    model: str = DEFAULT_MODEL, backend: str | None = None
) -> StressTestResult:
    """Test cross-field dependency constraints."""
    client = make_client(model, backend)

    cases = [
        (
            "date_range",
            lambda: client.ask_llm(
                "Create a date range spanning at least 2 years.",
                model=model,
                response_type=DateRange,
            ),
            validate_date_range,
        ),
        (
            "minmax_pair",
            lambda: client.ask_llm(
                "Create a min/max range with current value in between.",
                model=model,
                response_type=MinMaxPair,
            ),
            validate_minmax,
        ),
        (
            "age_relationships",
            lambda: client.ask_llm(
                "Create realistic ages for a three-generation family.",
                model=model,
                response_type=ParentChild,
            ),
            validate_ages,
        ),
        (
            "coordinate_distance",
            lambda: client.ask_llm(
                "Create origin and point coordinates with correct Euclidean distance.",
                model=model,
                response_type=Coordinates,
            ),
            validate_coordinates,
        ),
    ]

    return run_test_cases(client, "cross_field_deps", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
