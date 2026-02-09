"""Implementation tests for caller context tracking."""

from covenance._caller_context import (
    _caller_info_ctx,
    _get_caller_info_from_stack,
    capture_caller_context,
    get_caller_info,
)


def test_get_caller_info_from_stack_finds_external_frame():
    """Stack walker returns this test file as the external caller."""
    fn, file, line = _get_caller_info_from_stack()
    assert fn == "test_get_caller_info_from_stack_finds_external_frame"
    assert file == "test_caller_context.py"
    assert isinstance(line, int) and line > 0


def test_capture_and_get_caller_context():
    """capture_caller_context stores info that get_caller_info retrieves."""
    capture_caller_context()
    fn, file, line = get_caller_info()
    assert fn == "test_capture_and_get_caller_context"
    assert file == "test_caller_context.py"

    # Clean up context var
    _caller_info_ctx.set(None)


def test_get_caller_info_falls_back_to_stack_when_no_context():
    """Without context var set, get_caller_info walks the stack."""
    _caller_info_ctx.set(None)
    fn, file, line = get_caller_info()
    assert fn == "test_get_caller_info_falls_back_to_stack_when_no_context"
    assert file == "test_caller_context.py"


def test_context_var_takes_precedence_over_stack():
    """When context var is set, it takes precedence."""
    _caller_info_ctx.set(("custom_fn", "custom.py", 99))
    fn, file, line = get_caller_info()
    assert fn == "custom_fn"
    assert file == "custom.py"
    assert line == 99

    # Clean up
    _caller_info_ctx.set(None)
