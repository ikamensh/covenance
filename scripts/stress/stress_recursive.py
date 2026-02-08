"""Stress test: recursive/self-referencing types."""

from __future__ import annotations

from pydantic import BaseModel
from stress_utils import DEFAULT_MODEL, StressTestResult, run_test_cases

from covenance import Covenance


class TreeNode(BaseModel):
    """A tree node that can have children of the same type."""

    name: str
    children: list[TreeNode] = []


class LinkedItem(BaseModel):
    """A linked list node."""

    value: str
    next: LinkedItem | None = None


def count_tree_nodes(node: TreeNode) -> int:
    """Count total nodes in tree."""
    return 1 + sum(count_tree_nodes(c) for c in node.children)


def get_tree_depth(node: TreeNode) -> int:
    """Get maximum depth of tree."""
    if not node.children:
        return 1
    return 1 + max(get_tree_depth(c) for c in node.children)


def validate_tree(result: TreeNode, min_nodes: int, min_depth: int) -> tuple[bool, str]:
    """Validate tree has minimum structure."""
    nodes = count_tree_nodes(result)
    depth = get_tree_depth(result)

    issues = []
    if nodes < min_nodes:
        issues.append(f"nodes={nodes} < {min_nodes}")
    if depth < min_depth:
        issues.append(f"depth={depth} < {min_depth}")

    if issues:
        return False, f"Tree too small: {', '.join(issues)}"
    return True, ""


def validate_linked_list(result: LinkedItem, min_length: int) -> tuple[bool, str]:
    """Validate linked list has minimum length."""
    length = 0
    curr = result
    seen = set()
    while curr is not None:
        if id(curr) in seen:
            return False, "Circular reference detected"
        seen.add(id(curr))
        if not curr.value:
            return False, f"Empty value at position {length}"
        length += 1
        curr = curr.next
        if length > 100:  # Safety limit
            return False, "List too long (>100)"

    if length < min_length:
        return False, f"List length {length} < {min_length}"
    return True, ""


def run_stress_test(model: str = DEFAULT_MODEL) -> StressTestResult:
    """Test recursive type structures."""
    client = Covenance(label="stress_recursive")

    cases = [
        (
            "binary_tree",
            lambda: client.ask_llm(
                "Create a binary tree with exactly 7 nodes (root + 2 children each having 2 children). "
                "Each node needs a unique name.",
                model=model,
                response_type=TreeNode,
            ),
            lambda r: validate_tree(r, min_nodes=7, min_depth=3),
        ),
        (
            "linked_list",
            lambda: client.ask_llm(
                "Create a linked list of 5 items: apple -> banana -> cherry -> date -> elderberry",
                model=model,
                response_type=LinkedItem,
            ),
            lambda r: validate_linked_list(r, min_length=5),
        ),
    ]

    return run_test_cases(client, "recursive_types", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
