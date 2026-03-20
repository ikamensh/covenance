"""Implementation tests for Covenance client internals.

Tests internal implementation details like provider routing and backend selection.
These tests may break if the internal architecture changes.
"""

import pytest

from covenance import Covenance
from covenance.client import _NATIVE_BACKEND_PROVIDERS, _get_provider


def test_provider_routing():
    """Verify model names route to correct providers via _get_provider()."""
    cases = [
        ("gpt-5", "openai"),
        ("o3", "openai"),
        ("gemini-2.5-flash", "gemini"),
        ("claude-3.5-sonnet", "anthropic"),
        ("mistral-large", "mistral"),
        ("grok-4", "grok"),
        ("grok-4-fast", "grok"),
        ("grok-3-mini", "grok"),
        ("meta-llama/llama-3-70b", "openrouter"),
    ]
    for model, expected in cases:
        assert _get_provider(model) == expected, f"{model} should route to {expected}"


def test_native_backend_providers():
    """Verify native backend is used for OpenAI, Grok, and Mistral providers."""
    assert "openai" in _NATIVE_BACKEND_PROVIDERS
    assert "grok" in _NATIVE_BACKEND_PROVIDERS
    assert "mistral" in _NATIVE_BACKEND_PROVIDERS
    # pydantic-ai providers should NOT be in native backend
    assert "gemini" not in _NATIVE_BACKEND_PROVIDERS
    assert "anthropic" not in _NATIVE_BACKEND_PROVIDERS


def test_native_clients_lazy_creation():
    """Native clients are created lazily when first needed."""
    client = Covenance()

    # Initially no native clients created
    assert client._native_clients is None

    # After getting a native client, the dict is created
    native_client = client._get_native_client("openai")
    assert client._native_clients is not None
    assert "openai" in client._native_clients


def test_default_backends():
    """Default backends match expected routing."""
    client = Covenance()
    assert client.backends.openai == "native"
    assert client.backends.grok == "native"
    assert client.backends.gemini == "pydantic"
    assert client.backends.anthropic == "pydantic"
    assert client.backends.mistral == "native"
    assert client.backends.openrouter == "pydantic"


def test_backends_override_single():
    """Can override a single provider's backend after init."""
    client = Covenance()
    client.backends.anthropic = "native"
    assert client.backends.anthropic == "native"
    # Others unchanged
    assert client.backends.openai == "native"
    assert client.backends.gemini == "pydantic"


def test_backends_set_all():
    """set_all overrides every provider."""
    client = Covenance()
    client.backends.set_all("pydantic")
    assert client.backends.openai == "pydantic"
    assert client.backends.grok == "pydantic"
    assert client.backends.gemini == "pydantic"


def test_backends_rejects_invalid_backend():
    """Setting an invalid backend value raises ValueError."""
    client = Covenance()
    with pytest.raises(ValueError, match="Invalid backend"):
        client.backends.openai = "foo"


def test_backends_rejects_invalid_provider():
    """Setting an unknown provider raises AttributeError."""
    client = Covenance()
    with pytest.raises(AttributeError, match="No provider"):
        client.backends.unknown = "native"


def test_backends_repr():
    """Backends repr groups by backend type."""
    client = Covenance()
    r = repr(client.backends)
    assert "native=" in r
    assert "pydantic=" in r
    assert "openai" in r


def test_backends_get_unknown_provider():
    """get() returns 'pydantic' for unknown providers."""
    client = Covenance()
    assert client.backends.get("unknown_provider") == "pydantic"


def test_get_provider_canonical():
    """get_provider in _routing module works the same as the re-exported _get_provider."""
    from covenance._routing import get_provider

    assert get_provider("gpt-4") == "openai"
    assert get_provider("gemini-2.5-flash") == "gemini"
    assert get_provider("mistral-large") == "mistral"
    assert get_provider("claude-3.5-sonnet") == "anthropic"
    assert get_provider("grok-4") == "grok"
    assert get_provider("meta-llama/llama-3-70b") == "openrouter"
