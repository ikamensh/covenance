"""Tests for pricing calculation with cached tokens."""

from covenance.pricing import calculate_cost, get_model_pricing


def test_cached_tokens_reduce_cost():
    """Cached tokens should cost less than fresh input tokens."""
    model, provider = "gpt-5-mini", "openai"
    input_tokens = 10000
    output_tokens = 1000
    cached_tokens = 8000

    # Cost with no caching
    cost_no_cache = calculate_cost(
        model, provider, input_tokens, output_tokens, cached_tokens=0
    )

    # Cost with caching
    cost_with_cache = calculate_cost(
        model, provider, input_tokens, output_tokens, cached_tokens=cached_tokens
    )

    assert cost_with_cache < cost_no_cache, "Cached tokens should reduce total cost"

    # Verify the math: gpt-5-mini has input=$0.25, cached=$0.025, output=$2.00 per 1M
    pricing = get_model_pricing(model, provider)
    expected_fresh = (input_tokens - cached_tokens) / 1_000_000 * pricing.input
    expected_cached = cached_tokens / 1_000_000 * pricing.cached
    expected_output = output_tokens / 1_000_000 * pricing.output
    expected_total = round(expected_fresh + expected_cached + expected_output, 6)

    assert cost_with_cache == expected_total


def test_cached_tokens_subset_invariant():
    """Cost should be same whether we pass cached=0 or omit it."""
    cost_explicit = calculate_cost("gpt-5", "openai", 1000, 500, cached_tokens=0)
    cost_default = calculate_cost("gpt-5", "openai", 1000, 500)
    assert cost_explicit == cost_default


def test_model_without_caching_support():
    """Models without caching should charge full input price for 'cached' tokens."""
    # gemini-2.0-flash-lite has cached=None
    cost = calculate_cost(
        "gemini-2.0-flash-lite", "gemini", 10000, 1000, cached_tokens=5000
    )
    # Should use input price for all tokens since caching not supported
    pricing = get_model_pricing("gemini-2.0-flash-lite", "gemini")
    assert pricing.cached is None

    expected = (10000 / 1_000_000) * pricing.input + (1000 / 1_000_000) * pricing.output
    assert cost == round(expected, 6)


def test_unknown_model_returns_none():
    """Unknown models should return None for cost."""
    assert calculate_cost("unknown-model", "openai", 1000, 500) is None
    assert calculate_cost("gpt-5", "unknown-provider", 1000, 500) is None
