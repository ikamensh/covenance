"""Stress test: deeply nested type structures."""

from pydantic import BaseModel
from stress_utils import DEFAULT_MODEL, StressTestResult, run_test_cases

from covenance import Covenance


# Build nested models dynamically
class Level5(BaseModel):
    value: str


class Level4(BaseModel):
    name: str
    child: Level5


class Level3(BaseModel):
    name: str
    child: Level4


class Level2(BaseModel):
    name: str
    child: Level3


class Level1(BaseModel):
    name: str
    child: Level2


# Even deeper: 8 levels
class D8(BaseModel):
    val: str


class D7(BaseModel):
    n: str
    c: D8


class D6(BaseModel):
    n: str
    c: D7


class D5(BaseModel):
    n: str
    c: D6


class D4(BaseModel):
    n: str
    c: D5


class D3(BaseModel):
    n: str
    c: D4


class D2(BaseModel):
    n: str
    c: D3


class D1(BaseModel):
    n: str
    c: D2


def validate_depth_5(result: Level1) -> tuple[bool, str]:
    """Check all 5 levels are populated."""
    try:
        path = [
            result.name,
            result.child.name,
            result.child.child.name,
            result.child.child.child.name,
            result.child.child.child.child.value,
        ]
        if all(p and len(p) > 0 for p in path):
            return True, ""
        return False, f"Empty values in path: {path}"
    except AttributeError as e:
        return False, f"Missing nested field: {e}"


def validate_depth_8(result: D1) -> tuple[bool, str]:
    """Check all 8 levels are populated."""
    try:
        curr = result
        depth = 1
        while hasattr(curr, "c"):
            if not curr.n:
                return False, f"Empty name at depth {depth}"
            curr = curr.c
            depth += 1
        if not curr.val:
            return False, "Empty value at deepest level"
        if depth != 8:
            return False, f"Only reached depth {depth}, expected 8"
        return True, ""
    except AttributeError as e:
        return False, f"Missing nested field: {e}"


def run_stress_test(model: str = DEFAULT_MODEL) -> StressTestResult:
    """Test deeply nested structures."""
    client = Covenance(label="stress_nesting")

    cases = [
        (
            "depth_5",
            lambda: client.ask_llm(
                "Create a 5-level hierarchy: Company > Department > Team > Employee > Badge. "
                "Use realistic names.",
                model=model,
                response_type=Level1,
            ),
            validate_depth_5,
        ),
        (
            "depth_8",
            lambda: client.ask_llm(
                "Create an 8-level nested structure with names at each level.",
                model=model,
                response_type=D1,
            ),
            validate_depth_8,
        ),
    ]

    return run_test_cases(client, "nesting_depth", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
