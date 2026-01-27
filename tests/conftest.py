"""Pytest configuration and shared fixtures."""

from unittest.mock import patch

import pytest

# NOTE: Do NOT import covenance at module level!
# Imports must happen AFTER pytest_configure patches are active.

# Global storage for patches started in pytest_configure
_llm_patches: list[patch] = []

# Storage for original functions (saved before patching)
_original_llm_functions: dict[str, object] = {}


class LLMCallNotAllowedError(Exception):
    """Raised when a test attempts to make an actual LLM call."""


def _raise_llm_error(*args, **kwargs):
    raise LLMCallNotAllowedError(
        "Real LLM calls are not allowed in tests. "
        "Please mock LLM functions explicitly using unittest.mock.patch."
    )


def _online_tests_enabled(config: pytest.Config) -> bool:
    """Check if online tests are enabled via -m online flag."""
    markexpr = (config.option.markexpr or "").strip()
    # Check if markexpr is exactly "online" or contains "online" (for expressions like "online and not slow")
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


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_online = _online_tests_enabled(config)
    skip_reason = "run with: pytest -m online"
    for item in items:
        if item.get_closest_marker("online"):
            if not run_online:
                item.add_marker(pytest.mark.skip(reason=skip_reason))
            else:
                # Add unblock_llm fixture to online tests so they can make real calls
                # Use request.getfixturevalue to ensure it runs before the test
                item.add_marker(pytest.mark.usefixtures("unblock_llm"))


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


def pytest_unconfigure(config):
    """Clean up patches started in pytest_configure."""
    global _llm_patches
    for p in _llm_patches:
        try:
            p.stop()
        except Exception:
            pass
    _llm_patches = []
