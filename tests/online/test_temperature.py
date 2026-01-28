"""Online tests for temperature parameter.

Tests that temperature=0 produces deterministic output across multiple calls,
and that high temperature produces varied output.

Provider-specific notes:
- Anthropic: temperature range is 0..1 (not 0..2)
- OpenAI o-series/reasoning models: don't support temperature
- Grok: temp=0 may not be fully deterministic (observed in testing)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

import covenance
from covenance.models import ClaudeModels, GrokModels

pytestmark = pytest.mark.online

# Providers with models that support temperature
# Note: gpt-5-nano doesn't support temperature, using gpt-4.1-nano instead
PROVIDERS = [
    ("gpt-4.1-nano", "openai"),
    ("gemini-2.5-flash-lite", "gemini"),
    ("mistral-small-latest", "mistral"),
    (ClaudeModels.haiku45, "anthropic"),
    (GrokModels.grok3_mini, "grok"),
]

# Providers known to be deterministic at temp=0 in practice
# Grok excluded: observed non-determinism at temp=0 (2025-01).
# Grok's Mixture-of-Experts architecture may contribute to this.
DETERMINISTIC_PROVIDERS = [
    ("gpt-4.1-nano", "openai"),
    ("gemini-2.5-flash-lite", "gemini"),
    ("mistral-small-latest", "mistral"),
    (ClaudeModels.haiku45, "anthropic"),
]

# Question designed to invite diverse answers
CREATIVE_PROMPT = (
    "Name one unusual ice cream flavor. Just the flavor name, nothing else."
)

CALLS_PER_TEMP = 3

# Anthropic max is 1.0, others support up to 2.0 - use 0.95 for compatibility
HIGH_TEMPERATURE = 0.95


def _make_calls(model: str, temperature: float, num_calls: int) -> list[str]:
    """Make parallel calls with given temperature, return list of responses."""
    results = []
    with ThreadPoolExecutor(max_workers=num_calls) as executor:
        futures = [
            executor.submit(
                covenance.ask_llm,
                CREATIVE_PROMPT,
                model,
                temperature=temperature,
            )
            for _ in range(num_calls)
        ]
        for future in as_completed(futures):
            results.append(future.result().strip().lower())
    return results


@pytest.fixture(autouse=True)
def reset_records():
    covenance.clear_records()
    yield
    covenance.clear_records()


@pytest.mark.parametrize("model,provider", DETERMINISTIC_PROVIDERS)
def test_temperature_zero_deterministic(unblock_llm, model, provider):
    """Temperature=0 should produce identical outputs across concurrent calls."""
    results = _make_calls(model, temperature=0.0, num_calls=CALLS_PER_TEMP)
    unique_count = len(set(results))
    assert unique_count == 1, (
        f"{provider}: expected 1 unique output, got {unique_count}: {results}"
    )


def test_temperature_high_produces_variety(unblock_llm):
    """High temperature should produce varied outputs (soft expectation).

    This test uses xfail when no variety is found because high temperature
    doesn't guarantee diversity - it just increases the probability.
    """
    variety_found = {}

    def test_provider(model: str, provider: str) -> tuple[str, list[str], bool]:
        results = _make_calls(
            model, temperature=HIGH_TEMPERATURE, num_calls=CALLS_PER_TEMP
        )
        has_variety = len(set(results)) > 1
        return provider, results, has_variety

    # Run all providers in parallel
    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as executor:
        futures = {
            executor.submit(test_provider, model, provider): (model, provider)
            for model, provider in PROVIDERS
        }
        for future in as_completed(futures):
            model, provider = futures[future]
            prov_name, results, has_variety = future.result()
            variety_found[prov_name] = (has_variety, results)

    # Report findings
    providers_with_variety = [p for p, (v, _) in variety_found.items() if v]
    providers_without = [p for p, (v, _) in variety_found.items() if not v]

    print(
        f"\nProviders showing variety at temp={HIGH_TEMPERATURE}: {providers_with_variety}"
    )
    print(f"Providers with identical outputs: {providers_without}")
    for prov, (has_var, results) in variety_found.items():
        unique = len(set(results))
        print(f"  {prov}: {unique}/{len(results)} unique - {results}")

    # Soft assertion: at least some providers should show variety
    if not providers_with_variety:
        pytest.xfail("No providers showed variety at high temperature (not guaranteed)")
