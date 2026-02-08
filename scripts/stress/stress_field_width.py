"""Stress test: models with many fields (width)."""

from pydantic import BaseModel, create_model
from stress_utils import DEFAULT_MODEL, StressTestResult, run_test_cases

from covenance import Covenance


# 15 fields
class Person15(BaseModel):
    first_name: str
    last_name: str
    age: int
    email: str
    phone: str
    street: str
    city: str
    country: str
    occupation: str
    company: str
    salary: int
    is_employed: bool
    years_experience: int
    education_level: str
    favorite_color: str


# 30 fields - dynamically created
_fields_30 = {f"field_{i}": (str, ...) for i in range(30)}
Wide30 = create_model("Wide30", **_fields_30)

# 50 fields
_fields_50 = {f"f{i}": (str, ...) for i in range(50)}
Wide50 = create_model("Wide50", **_fields_50)


def validate_person15(result: Person15) -> tuple[bool, str]:
    """Check all 15 fields are populated."""
    data = result.model_dump()
    empty = [k for k, v in data.items() if v is None or v == ""]
    if empty:
        return False, f"Empty fields: {empty}"
    return True, ""


def validate_wide(result, expected_count: int) -> tuple[bool, str]:
    """Check all fields are populated."""
    data = result.model_dump()
    if len(data) != expected_count:
        return False, f"Expected {expected_count} fields, got {len(data)}"
    empty = [k for k, v in data.items() if v is None or v == ""]
    if empty:
        return False, f"Empty fields ({len(empty)}): {empty[:5]}..."
    return True, ""


def run_stress_test(model: str = DEFAULT_MODEL) -> StressTestResult:
    """Test wide models with many fields."""
    client = Covenance(label="stress_width")

    cases = [
        (
            "15_fields",
            lambda: client.ask_llm(
                "Generate a fictional person profile with all fields filled.",
                model=model,
                response_type=Person15,
            ),
            validate_person15,
        ),
        (
            "30_fields",
            lambda: client.ask_llm(
                "Fill all 30 string fields with unique single words.",
                model=model,
                response_type=Wide30,
            ),
            lambda r: validate_wide(r, 30),
        ),
        (
            "50_fields",
            lambda: client.ask_llm(
                "Fill all 50 string fields. Use single unique words for each.",
                model=model,
                response_type=Wide50,
            ),
            lambda r: validate_wide(r, 50),
        ),
    ]

    return run_test_cases(client, "field_width", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
