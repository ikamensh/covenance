"""Pytest configuration and shared fixtures."""

import hashlib
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# NOTE: Do NOT import covenance at module level!
# Imports must happen AFTER pytest_configure patches are active.

# Global storage for patches started in pytest_configure
_llm_patches: list[patch] = []

# Storage for original functions (saved before patching)
_original_llm_functions: dict[str, object] = {}


def _get_online_records_dir(config: pytest.Config) -> Path:
    """Deterministic temp dir for online test records (same path across xdist workers)."""
    # Use md5 for deterministic hash (Python's hash() is randomized per process)
    rootdir_hash = hashlib.md5(str(config.rootdir).encode()).hexdigest()[:8]
    return Path(tempfile.gettempdir()) / f"covenance_online_records_{rootdir_hash}"


class LLMCallNotAllowedError(Exception):
    """Raised when a test attempts to make an actual LLM call."""


def _raise_llm_error(*args, **kwargs):
    raise LLMCallNotAllowedError(
        "Real LLM calls are not allowed in tests. "
        "Please mock LLM functions explicitly using unittest.mock.patch."
    )


def pytest_addoption(parser):
    """Register --run-online flag to enable online tests without filtering."""
    parser.addoption(
        "--all",
        dest="run_all",
        action="store_true",
        default=False,
        help="Run all stable tests (offline + online)",
    )
    parser.addoption(
        "--run-online",
        action="store_true",
        default=False,
        help="Enable online tests (real API calls) without filtering to only online tests",
    )
    parser.addoption(
        "--run-unstable-external",
        action="store_true",
        default=False,
        help="Enable tests that depend on unreliable third-party APIs",
    )


def _online_tests_enabled(config: pytest.Config) -> bool:
    """Check if online tests are enabled via --run-online flag or -m online marker."""
    if config.option.run_all:
        return True
    # --run-online enables online tests without filtering
    if config.option.run_online:
        return True
    # -m online both enables and filters to online tests only
    markexpr = (config.option.markexpr or "").strip()
    return markexpr == "online" or (markexpr and "online" in markexpr.lower())


def pytest_configure(config):
    """Patch LLM wrappers before tests import the package."""
    global _llm_patches, _original_llm_functions

    import covenance.client as client_module

    _original_llm_functions["ask_llm"] = client_module.ask_llm
    _original_llm_functions["llm_consensus"] = client_module.llm_consensus

    _llm_patches = [
        patch("covenance.client.ask_llm", _raise_llm_error),
        patch("covenance.client.llm_consensus", _raise_llm_error),
        patch("covenance.ask_llm", _raise_llm_error),
        patch("covenance.llm_consensus", _raise_llm_error),
    ]
    for p in _llm_patches:
        p.start()

    # Set up records dir for online tests (xdist-compatible: deterministic path)
    if _online_tests_enabled(config):
        records_dir = _get_online_records_dir(config)
        records_dir.mkdir(parents=True, exist_ok=True)
        import covenance

        covenance.set_records_dir(records_dir)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_online = _online_tests_enabled(config)
    run_unstable = config.option.run_unstable_external
    # Use empty reason to avoid cluttering output - we'll print a single summary instead
    skip_reason = ""
    skipped_online_count = 0
    skipped_unstable_count = 0
    for item in items:
        # Skip unstable_external tests unless explicitly enabled
        if item.get_closest_marker("unstable_external") and not run_unstable:
            item.add_marker(pytest.mark.skip(reason=skip_reason))
            skipped_unstable_count += 1
            continue

        if item.get_closest_marker("online"):
            if not run_online:
                item.add_marker(pytest.mark.skip(reason=skip_reason))
                skipped_online_count += 1
            else:
                # Add unblock_llm fixture to online tests so they can make real calls
                # Use request.getfixturevalue to ensure it runs before the test
                item.add_marker(pytest.mark.usefixtures("unblock_llm"))
    # Store counts for terminal summary (since -rfE filters skipped from stats)
    config._skipped_online_count = skipped_online_count
    config._skipped_unstable_count = skipped_unstable_count


@pytest.fixture
def real_ask_llm():
    return _original_llm_functions["ask_llm"]


@pytest.fixture
def real_llm_consensus():
    return _original_llm_functions["llm_consensus"]


# Backwards compat aliases
@pytest.fixture
def real_ask_llm_structured(real_ask_llm):
    return real_ask_llm


@pytest.fixture
def real_ask_llm_structured_with_consensus(real_llm_consensus):
    return real_llm_consensus


@pytest.fixture
def clean_client_registry():
    """Clear client registry before and after each test to isolate test state."""
    import covenance.client as client_module

    # Store original registry contents
    original_clients = client_module._all_clients.copy()
    client_module._all_clients.clear()

    # Re-add only the default client
    client_module._all_clients.append(client_module._default_client)
    client_module._default_client.clear_records()

    yield

    # Restore original state
    client_module._all_clients.clear()
    client_module._all_clients.extend(original_clients)


@pytest.fixture
def unblock_llm():
    """Unblock LLM calls for online tests by stopping patches and restoring functions."""
    # Stop all patches first - this should restore the original functions
    for p in _llm_patches:
        p.stop()

    # Explicitly restore the original functions to ensure they're available
    import covenance as llm_init
    import covenance.client as client_module

    # Restore functions in client module
    client_module.ask_llm = _original_llm_functions["ask_llm"]
    client_module.llm_consensus = _original_llm_functions["llm_consensus"]

    # Restore functions in main module (re-exported from client)
    llm_init.ask_llm = _original_llm_functions["ask_llm"]
    llm_init.llm_consensus = _original_llm_functions["llm_consensus"]

    yield

    # Re-apply patches after test
    for p in _llm_patches:
        p.start()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print LLM cost summary after online tests and aggregated skip message."""
    # Only print on the main process (not xdist workers)
    if hasattr(config, "workerinput"):
        return

    # Print single summary line for skipped online tests
    skipped_online = getattr(config, "_skipped_online_count", 0)
    skipped_unstable = getattr(config, "_skipped_unstable_count", 0)
    if skipped_online > 0 or skipped_unstable > 0:
        terminalreporter.write_sep("=", "short test summary info")
        if skipped_online > 0:
            terminalreporter.write_line(
                f"SKIPPED [{skipped_online}] online tests (run with: pytest --all or pytest -m online)"
            )
        if skipped_unstable > 0:
            terminalreporter.write_line(
                f"SKIPPED [{skipped_unstable}] unstable_external tests (run with: --run-unstable-external)"
            )
    if not _online_tests_enabled(config):
        return

    # Print LLM cost summary for online tests
    records_file = _get_online_records_dir(config) / "llm_call_records.jsonl"
    if not records_file.exists():
        return

    from covenance.record import Record

    total_cost = 0.0
    total_input = 0
    total_output = 0
    count = 0
    models_used: set[str] = set()
    has_openrouter = False

    for line in records_file.read_text().strip().split("\n"):
        if not line:
            continue
        record = Record.model_validate_json(line)
        if record.cost_usd is not None:
            total_cost += record.cost_usd
        total_input += record.tokens_input
        total_output += record.tokens_output
        models_used.add(f"{record.provider}/{record.model}")
        if record.provider == "openrouter":
            has_openrouter = True
        count += 1

    if count == 0:
        return

    terminalreporter.write_sep("=", "LLM Cost Summary")
    terminalreporter.write_line(f"  Calls: {count}")
    terminalreporter.write_line(
        f"  Tokens: {total_input + total_output:,} (in={total_input:,}, out={total_output:,})"
    )
    cost_line = f"  Cost: ${total_cost:.6f}"
    if has_openrouter:
        cost_line += " (excluding OpenRouter calls)"
    terminalreporter.write_line(cost_line)
    terminalreporter.write_line(f"  Models: {', '.join(sorted(models_used))}")


def pytest_unconfigure(config):
    """Clean up patches and temp records dir."""
    global _llm_patches
    for p in _llm_patches:
        try:
            p.stop()
        except Exception:
            pass
    _llm_patches = []

    # Clean up records dir (only on main process, not xdist workers)
    if _online_tests_enabled(config) and not hasattr(config, "workerinput"):
        shutil.rmtree(_get_online_records_dir(config), ignore_errors=True)
