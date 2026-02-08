"""Stress test: optional field handling."""

from pydantic import BaseModel
from stress_utils import DEFAULT_MODEL, StressTestResult, run_test_cases

from covenance import Covenance


class ProfileWithOptionals(BaseModel):
    """Mix of required and optional fields."""

    # Required
    name: str
    email: str

    # Optional
    phone: str | None = None
    address: str | None = None
    bio: str | None = None
    website: str | None = None
    company: str | None = None


class PartialData(BaseModel):
    """Model where we control which optionals should be filled."""

    id: int
    title: str
    description: str | None = None
    author: str | None = None
    tags: list[str] | None = None
    rating: float | None = None


def validate_required_present(result: ProfileWithOptionals) -> tuple[bool, str]:
    """Validate required fields are always present."""
    if not result.name:
        return False, "name is required but empty"
    if not result.email:
        return False, "email is required but empty"
    return True, ""


def validate_some_optionals_filled(result: ProfileWithOptionals) -> tuple[bool, str]:
    """Validate that when asked, some optionals are filled."""
    if not result.name or not result.email:
        return False, "Required fields missing"

    optionals = [
        result.phone,
        result.address,
        result.bio,
        result.website,
        result.company,
    ]
    filled = sum(1 for o in optionals if o is not None)

    if filled < 3:
        return False, f"Expected at least 3 optionals filled, got {filled}"
    return True, ""


def validate_specific_optionals(
    result: PartialData, should_have: list[str]
) -> tuple[bool, str]:
    """Validate specific optional fields are filled."""
    data = result.model_dump()

    for field in should_have:
        if data.get(field) is None:
            return False, f"Expected {field} to be filled, but it's None"
    return True, ""


def run_stress_test(model: str = DEFAULT_MODEL) -> StressTestResult:
    """Test optional field handling."""
    client = Covenance(label="stress_optionals")

    cases = [
        (
            "required_only",
            lambda: client.ask_llm(
                "Create a minimal profile with just name and email. Leave everything else null.",
                model=model,
                response_type=ProfileWithOptionals,
            ),
            validate_required_present,
        ),
        (
            "fill_optionals",
            lambda: client.ask_llm(
                "Create a complete profile for John Doe with all available fields filled in.",
                model=model,
                response_type=ProfileWithOptionals,
            ),
            validate_some_optionals_filled,
        ),
        (
            "selective_optionals",
            lambda: client.ask_llm(
                "Create a data entry with id=1, title='Test', and include description and tags. "
                "Leave author and rating as null.",
                model=model,
                response_type=PartialData,
            ),
            lambda r: validate_specific_optionals(r, ["description", "tags"]),
        ),
    ]

    return run_test_cases(client, "optional_fields", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
