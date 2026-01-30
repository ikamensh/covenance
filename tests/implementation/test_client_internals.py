"""Implementation tests for Covenance client internals.

Tests internal implementation details like lazy client creation, provider routing,
and _clients dict structure. These tests may break if the internal architecture changes.
"""

from covenance import Covenance

ˆc
def test_provider_routing():
    """Verify model names route to correct providers via _get_provider()."""
    client = Covenance()
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
        assert client._get_provider(model) == expected, (
            f"{model} should route to {expected}"
        )


def test_explicit_key_triggers_immediate_client_creation():
    """Explicit API keys trigger immediate SDK client creation (not lazy).

    When a key is provided explicitly at init, the client validates it by
    creating the SDK client immediately rather than deferring creation.
    """
    client = Covenance(openai_api_key="sk-test-key-for-validation")

    # The OpenAI client should already be instantiated (not lazy)
    openai_lazy = client._clients["openai"]
    assert openai_lazy._client is not None, (
        "Explicit key should trigger immediate client creation"
    )

    # Other providers should still be lazy (no explicit key)
    anthropic_lazy = client._clients["anthropic"]
    assert anthropic_lazy._client is None, "No explicit key means client stays lazy"
