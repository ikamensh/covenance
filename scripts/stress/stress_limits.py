"""Stress test: find breaking points and provider-specific limits.

Known limitations discovered:
- dict[str, str] NOT supported by any provider (OpenAI, Gemini, Mistral)
- Gemini: ~40-44 fields × 20-value Literal is the limit (schema complexity)
- Gemini: Recursive/self-referencing types cause RecursionError
- Gemini: additionalProperties not supported (affects dict types)
- Mistral/OpenAI: Handle 150+ Literal fields fine
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, create_model
from stress_utils import DEFAULT_MODEL, StressTestResult, safe_call

from covenance import Covenance

# Large Literal type (20 values)
StatusLiteral = Literal[
    "pending",
    "active",
    "inactive",
    "suspended",
    "cancelled",
    "completed",
    "failed",
    "timeout",
    "retry",
    "queued",
    "processing",
    "validated",
    "rejected",
    "approved",
    "archived",
    "deleted",
    "restored",
    "migrated",
    "synced",
    "error",
]


# Large Enum (20 values)
class StatusEnum(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRY = "retry"
    QUEUED = "queued"
    PROCESSING = "processing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    APPROVED = "approved"
    ARCHIVED = "archived"
    DELETED = "deleted"
    RESTORED = "restored"
    MIGRATED = "migrated"
    SYNCED = "synced"
    ERROR = "error"


# Test 1: 20-value Literal in 30 fields (should pass all)
_literal_fields_30 = {f"status_{i}": (StatusLiteral, ...) for i in range(30)}
ManyLiterals30 = create_model("ManyLiterals30", **_literal_fields_30)

# Test 2: 20-value Literal in 50 fields (Gemini fails here)
_literal_fields_50 = {f"status_{i}": (StatusLiteral, ...) for i in range(50)}
ManyLiterals50 = create_model("ManyLiterals50", **_literal_fields_50)

# Test 3: 20-value Literal in 100 fields (pushes limits)
_literal_fields_100 = {f"status_{i}": (StatusLiteral, ...) for i in range(100)}
ManyLiterals100 = create_model("ManyLiterals100", **_literal_fields_100)

# Test 4: 20-value Enum in 40 fields (near Gemini's limit)
_enum_fields_40 = {f"state_{i}": (StatusEnum, ...) for i in range(40)}
ManyEnums40 = create_model("ManyEnums40", **_enum_fields_40)

# Test 5: 150 string fields (very wide)
_wide_150 = {f"f{i}": (str, ...) for i in range(150)}
Wide150 = create_model("Wide150", **_wide_150)


# Test 6: 12-level nesting (deep)
class N12(BaseModel):
    v: str


class N11(BaseModel):
    n: str
    c: N12


class N10(BaseModel):
    n: str
    c: N11


class N9(BaseModel):
    n: str
    c: N10


class N8(BaseModel):
    n: str
    c: N9


class N7(BaseModel):
    n: str
    c: N8


class N6(BaseModel):
    n: str
    c: N7


class N5(BaseModel):
    n: str
    c: N6


class N4(BaseModel):
    n: str
    c: N5


class N3(BaseModel):
    n: str
    c: N4


class N2(BaseModel):
    n: str
    c: N3


class N1(BaseModel):
    n: str
    c: N2


# Test 7: Complex nested with enums (no dict - dict not supported anywhere)
class ComplexItem(BaseModel):
    id: int
    status: StatusEnum
    priority: Literal["low", "medium", "high", "critical", "urgent"]
    tags: list[str]


class ComplexContainer(BaseModel):
    """Complex nested structure - uses list of KV pairs instead of dict."""

    class KeyValue(BaseModel):
        key: str
        value: str

    name: str
    items: list[ComplexItem]
    metadata: list[KeyValue]  # Workaround for dict[str, str]


# Test 8: Many optional Literals (40 fields)
_optional_literals = {f"opt_{i}": (StatusLiteral | None, None) for i in range(40)}
ManyOptionalLiterals = create_model("ManyOptionalLiterals", **_optional_literals)


# Test 9: Many Literal values (50 values, 20 fields) - tests value count vs field count
BigLiteral = Literal[tuple(f"val_{i}" for i in range(50))]  # type: ignore
_big_literal_fields = {f"f{i}": (BigLiteral, ...) for i in range(20)}
ManyValues50 = create_model("ManyValues50", **_big_literal_fields)


def run_stress_test(model: str = DEFAULT_MODEL) -> StressTestResult:
    """Find breaking points with extreme schemas."""
    client = Covenance(label="stress_limits")
    failures = []
    details = {}

    tests = [
        # (name, type, prompt, validator)
        (
            "literal_30x20",
            ManyLiterals30,
            "Generate 30 different status values. Use variety.",
            lambda r: (True, "") if r else (False, "No result"),
        ),
        (
            "literal_50x20",
            ManyLiterals50,
            "Generate 50 different status values. Use variety.",
            lambda r: (True, "") if r else (False, "No result"),
        ),
        (
            "literal_100x20",
            ManyLiterals100,
            "Generate 100 status values. Vary them.",
            lambda r: (True, "") if r else (False, "No result"),
        ),
        (
            "enum_40x20",
            ManyEnums40,
            "Generate 40 state values using different enum options.",
            lambda r: (True, "") if r else (False, "No result"),
        ),
        (
            "width_150_fields",
            Wide150,
            "Fill all 150 fields with unique single words.",
            lambda r: validate_width(r, 150),
        ),
        (
            "depth_12_levels",
            N1,
            "Create a 12-level nested structure with names at each level.",
            lambda r: validate_depth(r, 12),
        ),
        (
            "complex_nested",
            ComplexContainer,
            "Create a container with 5 items (each with status, priority, tags) and 3 metadata key-value pairs.",
            lambda r: validate_complex(r, 5),
        ),
        (
            "optional_40x20",
            ManyOptionalLiterals,
            "Fill about half of the optional status fields, leave others null.",
            lambda r: (True, "") if r else (False, "No result"),
        ),
        (
            "values_50x20",
            ManyValues50,
            "Generate 20 fields, each picking from 50 possible values.",
            lambda r: (True, "") if r else (False, "No result"),
        ),
    ]

    for name, response_type, prompt, validator in tests:
        result, error = safe_call(
            client,
            lambda rt=response_type, p=prompt: client.ask_llm(
                p,
                model=model,
                response_type=rt,
            ),
            timeout_seconds=120,  # Longer timeout for complex schemas
        )

        if error:
            failures.append(f"{name}: {error}")
            details[name] = {"error": error}
        else:
            passed, reason = validator(result)
            if not passed:
                failures.append(f"{name}: {reason}")
                details[name] = {"passed": False, "reason": reason}
            else:
                details[name] = {"passed": True}

    summary = client.usage_summary()

    return StressTestResult(
        name="limits",
        passed=len(failures) == 0,
        failures=failures,
        calls_made=summary["calls"],
        cost_usd=summary["cost_usd"],
        details=details,
    )


def validate_width(result, expected: int) -> tuple[bool, str]:
    """Validate wide model has all fields."""
    if result is None:
        return False, "No result"
    data = result.model_dump()
    if len(data) != expected:
        return False, f"Expected {expected} fields, got {len(data)}"
    empty = sum(1 for v in data.values() if not v)
    if empty > expected * 0.1:  # Allow 10% empty
        return False, f"Too many empty fields: {empty}/{expected}"
    return True, ""


def validate_depth(result, expected: int) -> tuple[bool, str]:
    """Validate deep nesting reaches expected depth."""
    if result is None:
        return False, "No result"
    depth = 1
    curr = result
    try:
        while hasattr(curr, "c"):
            if not curr.n:
                return False, f"Empty name at depth {depth}"
            curr = curr.c
            depth += 1
        if not hasattr(curr, "v") or not curr.v:
            return False, "Missing value at deepest level"
    except Exception as e:
        return False, f"Error traversing: {e}"

    if depth != expected:
        return False, f"Reached depth {depth}, expected {expected}"
    return True, ""


def validate_complex(result, min_items: int) -> tuple[bool, str]:
    """Validate complex nested structure."""
    if result is None:
        return False, "No result"
    if not result.items or len(result.items) < min_items:
        return (
            False,
            f"Expected {min_items} items, got {len(result.items) if result.items else 0}",
        )
    for i, item in enumerate(result.items):
        if item.status not in StatusEnum:
            return False, f"Item {i} has invalid status"
    if not result.metadata:
        return False, "No metadata"
    return True, ""


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    result = run_stress_test(model)
    print(f"\n{'PASS' if result.passed else 'FAIL'}: {result.name}")
    print(f"Calls: {result.calls_made}, Cost: ${result.cost_usd:.4f}")
    for f in result.failures:
        print(f"  - {f}")
