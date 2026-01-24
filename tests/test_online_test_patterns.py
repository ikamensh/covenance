"""Test to ensure online tests follow the correct import pattern.

Online tests must use `import covenance` and then `covenance.ask_llm_structured()`
instead of `from covenance import ask_llm_structured` to ensure the unblock_llm
fixture can properly restore the functions.
"""

import re
from pathlib import Path

import pytest


def find_online_test_files() -> list[Path]:
    """Find all test files that are online tests."""
    tests_dir = Path(__file__).parent
    current_file = Path(__file__)
    online_tests = []

    # Find files in tests/online/ directory
    online_dir = tests_dir / "online"
    if online_dir.exists():
        online_tests.extend(online_dir.glob("test_*.py"))

    # Find files marked with pytest.mark.online
    for test_file in tests_dir.rglob("test_*.py"):
        if test_file.is_file() and test_file != current_file:
            content = test_file.read_text()
            # Check if file has pytestmark = pytest.mark.online or @pytest.mark.online
            if (
                "pytestmark = pytest.mark.online" in content
                or "@pytest.mark.online" in content
            ):
                online_tests.append(test_file)

    return list(set(online_tests))  # Remove duplicates


def check_for_bad_imports(file_path: Path) -> list[tuple[int, str]]:
    """Check a file for bad import patterns.

    Returns list of (line_number, line_content) tuples for violations.
    """
    violations = []
    content = file_path.read_text()
    lines = content.splitlines()

    # Pattern to match: from covenance import ask_llm_structured[...]
    # This matches:
    # - from covenance import ask_llm_structured
    # - from covenance import ask_llm_structured_with_consensus
    # - from covenance import ask_llm_structured, other_function
    # - from covenance import (ask_llm_structured)
    # - from covenance import ask_llm_structured as alias
    bad_pattern = re.compile(
        r"^\s*from\s+covenance\s+import\s+.*\b(ask_llm_structured|ask_llm_structured_with_consensus)\b"
    )

    for line_num, line in enumerate(lines, start=1):
        if bad_pattern.match(line):
            violations.append((line_num, line.strip()))

    return violations


def test_online_tests_use_correct_import_pattern():
    """Ensure online tests use `import covenance` instead of direct imports.

    Online tests must use:
        import covenance
        covenance.ask_llm_structured(...)

    NOT:
        from covenance import ask_llm_structured
        ask_llm_structured(...)

    This is required because the unblock_llm fixture restores functions on the
    covenance module object. When you use `from covenance import ask_llm_structured`,
    Python binds the function to a local name at import time, which won't be updated
    when the fixture restores the functions. Using `covenance.ask_llm_structured`
    performs attribute lookup at runtime, ensuring you get the restored function.
    """
    online_test_files = find_online_test_files()

    if not online_test_files:
        pytest.skip("No online test files found")

    all_violations = []
    for test_file in online_test_files:
        violations = check_for_bad_imports(test_file)
        if violations:
            all_violations.append((test_file, violations))

    if all_violations:
        error_msg = (
            "Online tests must use 'import covenance' and 'covenance.ask_llm_structured()' "
            "instead of 'from covenance import ask_llm_structured'.\n\n"
            "Reason: The unblock_llm fixture restores functions on the covenance module object. "
            "Direct imports bind the function at import time and won't see the restored function. "
            "Using 'covenance.ask_llm_structured' performs attribute lookup at runtime.\n\n"
            "Violations found:\n"
        )

        for file_path, violations in all_violations:
            error_msg += f"\n  {file_path.relative_to(Path(__file__).parent.parent)}:\n"
            for line_num, line_content in violations:
                error_msg += f"    Line {line_num}: {line_content}\n"

        error_msg += (
            "\nFix: Change from:\n"
            "    from covenance import ask_llm_structured\n"
            "    result = ask_llm_structured(...)\n\n"
            "To:\n"
            "    import covenance\n"
            "    result = covenance.ask_llm_structured(...)\n"
        )

        pytest.fail(error_msg)
