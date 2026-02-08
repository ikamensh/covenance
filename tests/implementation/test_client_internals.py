"""Implementation tests for Covenance client internals.

Tests internal implementation details like provider routing and backend selection.
These tests may break if the internal architecture changes.
"""

from covenance import Covenance
from covenance.client import _get_provider, _NATIVE_BACKEND_PROVIDERS


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
    """Verify native backend is used for OpenAI and Grok providers."""
    assert "openai" in _NATIVE_BACKEND_PROVIDERS
    assert "grok" in _NATIVE_BACKEND_PROVIDERS
    # pydantic-ai providers should NOT be in native backend
    assert "gemini" not in _NATIVE_BACKEND_PROVIDERS
    assert "anthropic" not in _NATIVE_BACKEND_PROVIDERS
    assert "mistral" not in _NATIVE_BACKEND_PROVIDERS


def test_native_clients_lazy_creation():
    """Native clients are created lazily when first needed."""
    client = Covenance()
    
    # Initially no native clients created
    assert client._native_clients is None
    
    # After getting a native client, the dict is created
    native_client = client._get_native_client("openai")
    assert client._native_clients is not None
    assert "openai" in client._native_clients
