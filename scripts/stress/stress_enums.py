"""Stress test: enum value adherence."""

from enum import StrEnum

from pydantic import BaseModel
from stress_utils import DEFAULT_MODEL, StressTestResult, make_client, run_test_cases


class Priority(StrEnum):
    """Uncommon enum values to test adherence."""

    URGENT_ALPHA = "urgent_alpha"
    STANDARD_BETA = "standard_beta"
    DEFERRED_GAMMA = "deferred_gamma"


class StatusCode(StrEnum):
    """Status codes that look like they could be anything."""

    X7_PENDING = "X7_PENDING"
    K3_APPROVED = "K3_APPROVED"
    M9_REJECTED = "M9_REJECTED"
    Z1_ARCHIVED = "Z1_ARCHIVED"


class TaskWithEnum(BaseModel):
    title: str
    priority: Priority
    status: StatusCode


class MultiEnumResult(BaseModel):
    items: list[TaskWithEnum]


def validate_single_enum(result: TaskWithEnum) -> tuple[bool, str]:
    """Validate enum values are valid."""
    # Pydantic already validates, but let's be explicit
    if result.priority not in Priority:
        return False, f"Invalid priority: {result.priority}"
    if result.status not in StatusCode:
        return False, f"Invalid status: {result.status}"
    return True, ""


def validate_multi_enum(
    result: MultiEnumResult, expected_count: int
) -> tuple[bool, str]:
    """Validate list of items with enums."""
    if len(result.items) < expected_count:
        return False, f"Expected {expected_count} items, got {len(result.items)}"

    for i, item in enumerate(result.items):
        if item.priority not in Priority:
            return False, f"Item {i}: invalid priority {item.priority}"
        if item.status not in StatusCode:
            return False, f"Item {i}: invalid status {item.status}"
    return True, ""


def validate_specific_enum(
    result: TaskWithEnum, expected_priority: Priority
) -> tuple[bool, str]:
    """Validate a specific enum value was chosen."""
    if result.priority != expected_priority:
        return False, f"Expected {expected_priority}, got {result.priority}"
    return True, ""


def run_stress_test(
    model: str = DEFAULT_MODEL, backend: str | None = None
) -> StressTestResult:
    """Test enum value adherence."""
    client = make_client(model, backend)

    cases = [
        (
            "basic_enum",
            lambda: client.ask_llm(
                "Create a task titled 'Review code'. Pick appropriate priority and status.",
                model=model,
                response_type=TaskWithEnum,
            ),
            validate_single_enum,
        ),
        (
            "specific_enum_value",
            lambda: client.ask_llm(
                "Create a task with DEFERRED_GAMMA priority (this is critical - use exactly this priority value).",
                model=model,
                response_type=TaskWithEnum,
            ),
            lambda r: validate_specific_enum(r, Priority.DEFERRED_GAMMA),
        ),
        (
            "multiple_enums",
            lambda: client.ask_llm(
                "Create 4 tasks with different combinations of priority and status values.",
                model=model,
                response_type=MultiEnumResult,
            ),
            lambda r: validate_multi_enum(r, 4),
        ),
    ]

    return run_test_cases(client, "enum_adherence", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
