"""Stress test: edge case values."""

from pydantic import BaseModel, Field
from stress_utils import DEFAULT_MODEL, StressTestResult, run_test_cases

from covenance import Covenance


class EdgeNumbers(BaseModel):
    """Numbers at various extremes."""

    zero: int = Field(description="Must be exactly 0")
    negative: int = Field(description="Must be a negative integer")
    large: int = Field(description="A large positive integer over 1 million")
    small_float: float = Field(description="A small positive decimal like 0.001")


class EdgeStrings(BaseModel):
    """Strings with special characteristics."""

    empty: str = Field(description="Must be an empty string ''")
    single_char: str = Field(description="Exactly one character")
    with_spaces: str = Field(description="A phrase with multiple spaces")
    unicode_text: str = Field(
        description="Text containing non-ASCII characters (e.g., émojis, accents)"
    )


class EdgeCollections(BaseModel):
    """Collections at boundaries."""

    empty_list: list[str] = Field(description="An empty list []")
    single_item: list[int] = Field(description="A list with exactly one item")
    nested_empty: list[list[str]] = Field(
        description="A list containing one empty list"
    )


class MixedEdges(BaseModel):
    """Combination of edge cases."""

    maybe_zero: int | None = Field(description="Either 0 or null")
    bool_as_int: int = Field(description="The integer 1 (representing true)")
    negative_float: float = Field(description="A negative decimal number")


def validate_edge_numbers(result: EdgeNumbers) -> tuple[bool, str]:
    """Validate numeric edge cases."""
    issues = []

    if result.zero != 0:
        issues.append(f"zero is {result.zero}, expected 0")
    if result.negative >= 0:
        issues.append(f"negative is {result.negative}, expected < 0")
    if result.large <= 1_000_000:
        issues.append(f"large is {result.large}, expected > 1M")
    if result.small_float <= 0 or result.small_float >= 1:
        issues.append(f"small_float is {result.small_float}, expected in (0, 1)")

    if issues:
        return False, f"Number edge violations: {issues}"
    return True, ""


def validate_edge_strings(result: EdgeStrings) -> tuple[bool, str]:
    """Validate string edge cases."""
    issues = []

    if result.empty != "":
        issues.append(f"empty is '{result.empty}', expected ''")
    if len(result.single_char) != 1:
        issues.append(
            f"single_char '{result.single_char}' has length {len(result.single_char)}"
        )
    if " " not in result.with_spaces or len(result.with_spaces.split()) < 2:
        issues.append(f"with_spaces '{result.with_spaces}' doesn't have multiple words")

    # Check for non-ASCII
    if result.unicode_text.isascii():
        issues.append(f"unicode_text '{result.unicode_text}' is ASCII-only")

    if issues:
        return False, f"String edge violations: {issues}"
    return True, ""


def validate_edge_collections(result: EdgeCollections) -> tuple[bool, str]:
    """Validate collection edge cases."""
    issues = []

    if result.empty_list != []:
        issues.append(f"empty_list is {result.empty_list}, expected []")
    if len(result.single_item) != 1:
        issues.append(f"single_item has {len(result.single_item)} items, expected 1")
    if not result.nested_empty or result.nested_empty[0] != []:
        issues.append(f"nested_empty is {result.nested_empty}, expected [[]]")

    if issues:
        return False, f"Collection edge violations: {issues}"
    return True, ""


def validate_mixed_edges(result: MixedEdges) -> tuple[bool, str]:
    """Validate mixed edge cases."""
    issues = []

    if result.maybe_zero is not None and result.maybe_zero != 0:
        issues.append(f"maybe_zero is {result.maybe_zero}, expected 0 or None")
    if result.bool_as_int != 1:
        issues.append(f"bool_as_int is {result.bool_as_int}, expected 1")
    if result.negative_float >= 0:
        issues.append(f"negative_float is {result.negative_float}, expected < 0")

    if issues:
        return False, f"Mixed edge violations: {issues}"
    return True, ""


def run_stress_test(model: str = DEFAULT_MODEL) -> StressTestResult:
    """Test edge case value handling."""
    client = Covenance(label="stress_edges")

    cases = [
        (
            "edge_numbers",
            lambda: client.ask_llm(
                "Generate the exact numeric values described in each field.",
                model=model,
                response_type=EdgeNumbers,
            ),
            validate_edge_numbers,
        ),
        (
            "edge_strings",
            lambda: client.ask_llm(
                "Generate strings matching exactly what each field description specifies.",
                model=model,
                response_type=EdgeStrings,
            ),
            validate_edge_strings,
        ),
        (
            "edge_collections",
            lambda: client.ask_llm(
                "Generate collections with the exact structure described.",
                model=model,
                response_type=EdgeCollections,
            ),
            validate_edge_collections,
        ),
        (
            "mixed_edges",
            lambda: client.ask_llm(
                "Generate values matching the edge case descriptions.",
                model=model,
                response_type=MixedEdges,
            ),
            validate_mixed_edges,
        ),
    ]

    return run_test_cases(client, "edge_values", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
