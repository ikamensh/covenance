"""Stress test: diverse type combinations."""

from pydantic import BaseModel
from stress_utils import DEFAULT_MODEL, StressTestResult, make_client, run_test_cases


class AllPrimitives(BaseModel):
    """All primitive types in one model."""

    string_val: str
    int_val: int
    float_val: float
    bool_val: bool


class WithCollections(BaseModel):
    """Model with collection types."""

    tags: list[str]
    scores: list[int]
    matrix: list[list[int]]


class NestedComplex(BaseModel):
    """Complex nested structure with mixed types."""

    class Inner(BaseModel):
        x: int
        y: float
        label: str

    class Metadata(BaseModel):
        key1: str
        key2: str

    name: str
    items: list[Inner]
    active: bool
    metadata: Metadata


def validate_primitives(result: AllPrimitives) -> tuple[bool, str]:
    """Validate all primitives are correct types."""
    issues = []
    if not isinstance(result.string_val, str) or not result.string_val:
        issues.append("string_val invalid")
    if not isinstance(result.int_val, int):
        issues.append("int_val not int")
    if not isinstance(result.float_val, float):
        issues.append("float_val not float")
    if not isinstance(result.bool_val, bool):
        issues.append("bool_val not bool")

    if issues:
        return False, f"Type issues: {issues}"
    return True, ""


def validate_collections(result: WithCollections) -> tuple[bool, str]:
    """Validate collections are populated correctly."""
    issues = []
    if not result.tags or len(result.tags) < 2:
        issues.append(f"tags too short: {result.tags}")
    if not result.scores or len(result.scores) < 2:
        issues.append(f"scores too short: {result.scores}")
    if not result.matrix or len(result.matrix) < 2:
        issues.append(f"matrix too small: {result.matrix}")
    else:
        for row in result.matrix:
            if not row or len(row) < 2:
                issues.append(f"matrix row too short: {row}")
                break

    if issues:
        return False, f"Collection issues: {issues}"
    return True, ""


def validate_nested_complex(result: NestedComplex) -> tuple[bool, str]:
    """Validate complex nested structure."""
    issues = []
    if not result.name:
        issues.append("name empty")
    if not result.items or len(result.items) < 2:
        issues.append(f"items too short: {len(result.items) if result.items else 0}")
    if not result.metadata.key1 or not result.metadata.key2:
        issues.append("metadata keys empty")

    if issues:
        return False, f"Nested issues: {issues}"
    return True, ""


def run_stress_test(
    model: str = DEFAULT_MODEL, backend: str | None = None
) -> StressTestResult:
    """Test type diversity handling."""
    client = make_client(model, backend)

    cases = [
        (
            "all_primitives",
            lambda: client.ask_llm(
                "Generate values: a greeting string, the number 42, pi to 2 decimals, and true.",
                model=model,
                response_type=AllPrimitives,
            ),
            validate_primitives,
        ),
        (
            "collections",
            lambda: client.ask_llm(
                "Generate: 3 color names as tags, 4 test scores (0-100), and a 3x3 matrix of small integers.",
                model=model,
                response_type=WithCollections,
            ),
            validate_collections,
        ),
        (
            "nested_complex",
            lambda: client.ask_llm(
                "Create a dataset named 'measurements' with 3 coordinate items (x,y,label), "
                "active=true, and metadata with 2 key-value pairs.",
                model=model,
                response_type=NestedComplex,
            ),
            validate_nested_complex,
        ),
    ]

    return run_test_cases(client, "type_diversity", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
