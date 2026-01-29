"""Verify local pricing data against external sources.

Data sources:
- LiteLLM: https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
- PricePerToken: https://pricepertoken.com (via MCP API)
"""

import json
import urllib.request

import pytest

from covenance.models.google import GEMINI_PRICING
from covenance.models.grok import GROK_PRICING
from covenance.models.openai import OPENAI_PRICING

PRICING = {
    "openai": OPENAI_PRICING,
    "gemini": GEMINI_PRICING,
    "grok": GROK_PRICING,
}

pytestmark = pytest.mark.online

LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

PRICEPERTOKEN_MCP_URL = "https://api.pricepertoken.com/mcp/mcp"

# Map our model names to LiteLLM's naming convention
# LiteLLM uses "xai/" prefix
MODEL_NAME_MAP = {
    # Grok models (xAI uses dashes: grok-4-1-fast, not grok-4.1-fast)
    ("grok", "grok-4"): "xai/grok-4",
    ("grok", "grok-4-1-fast"): "xai/grok-4-1-fast",
    ("grok", "grok-4-fast"): "xai/grok-4-fast-reasoning",
    ("grok", "grok-4-1-fast-non-reasoning"): "xai/grok-4-1-fast-non-reasoning",
    ("grok", "grok-4-fast-non-reasoning"): "xai/grok-4-fast-non-reasoning",
    ("grok", "grok-code-fast-1"): "xai/grok-code-fast-1",
    ("grok", "grok-3"): "xai/grok-3",
    ("grok", "grok-3-mini"): "xai/grok-3-mini",
    # OpenAI models
    ("openai", "gpt-5"): "gpt-5",
    ("openai", "gpt-5-mini"): "gpt-5-mini",
    ("openai", "gpt-5-nano"): "gpt-5-nano",
    ("openai", "gpt-5.1"): "gpt-5.1",
    ("openai", "gpt-5.2"): "gpt-5.2",
    ("openai", "o3"): "o3",
    ("openai", "o4-mini"): "o4-mini",
    # Gemini models (use non-prefixed names which are stable/GA)
    ("gemini", "gemini-2.5-pro"): "gemini-2.5-pro",
    ("gemini", "gemini-2.5-flash"): "gemini-2.5-flash",
    ("gemini", "gemini-2.5-flash-lite"): "gemini-2.5-flash-lite",
    ("gemini", "gemini-2.0-flash"): "gemini-2.0-flash",
    ("gemini", "gemini-2.0-flash-lite"): "gemini-2.0-flash-lite",
}

# Tolerance for floating point comparison (5% relative tolerance)
PRICE_TOLERANCE = 0.05


def fetch_litellm_pricing() -> dict:
    """Fetch and parse LiteLLM pricing JSON."""
    with urllib.request.urlopen(LITELLM_PRICING_URL, timeout=30) as response:
        return json.loads(response.read().decode())


def fetch_pricepertoken_models(author: str) -> list[dict]:
    """Fetch models from pricepertoken.com MCP API.

    Their server blocks python-httpx User-Agent, so we use urllib with custom UA.
    """
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "get_all_models", "arguments": {"author": author}},
        }
    ).encode()

    request = urllib.request.Request(
        PRICEPERTOKEN_MCP_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "covenance-test/1.0",  # Required: they block python-httpx
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        resp = json.loads(response.read().decode())
        # MCP returns nested JSON: result.content[0].text contains the actual data
        return json.loads(resp["result"]["content"][0]["text"])


def to_per_million(cost_per_token: float | None) -> float | None:
    """Convert cost per token to cost per million tokens."""
    if cost_per_token is None:
        return None
    return cost_per_token * 1_000_000


def prices_match(
    ours: float, theirs: float, tolerance: float = PRICE_TOLERANCE
) -> bool:
    """Check if prices match within tolerance."""
    if ours == theirs == 0:
        return True
    if ours == 0 or theirs == 0:
        return False
    return abs(ours - theirs) / max(ours, theirs) <= tolerance


@pytest.fixture(scope="module")
def litellm_pricing():
    """Fetch LiteLLM pricing once per test module."""
    return fetch_litellm_pricing()


@pytest.fixture(scope="module")
def pricepertoken_xai():
    """Fetch xAI/Grok pricing from pricepertoken.com once per test module."""
    return fetch_pricepertoken_models("xai")


def test_pricing_against_litellm(litellm_pricing):
    """Verify our pricing matches LiteLLM's data.

    This test compares input, output, and cached token prices for models
    where we have a mapping to LiteLLM's naming convention.
    """
    mismatches = []
    missing = []

    for (provider, model), litellm_name in MODEL_NAME_MAP.items():
        if provider not in PRICING:
            continue
        if model not in PRICING[provider]:
            continue

        our_pricing = PRICING[provider][model]

        if litellm_name not in litellm_pricing:
            missing.append(f"{provider}/{model} -> {litellm_name}")
            continue

        their_pricing = litellm_pricing[litellm_name]
        their_input = to_per_million(their_pricing.get("input_cost_per_token"))
        their_output = to_per_million(their_pricing.get("output_cost_per_token"))
        their_cached = to_per_million(their_pricing.get("cache_read_input_token_cost"))

        errors = []
        if their_input and not prices_match(our_pricing.input, their_input):
            errors.append(
                f"input: ours=${our_pricing.input:.3f} vs theirs=${their_input:.3f}"
            )
        if their_output and not prices_match(our_pricing.output, their_output):
            errors.append(
                f"output: ours=${our_pricing.output:.3f} vs theirs=${their_output:.3f}"
            )
        if their_cached and our_pricing.cached:
            if not prices_match(our_pricing.cached, their_cached):
                errors.append(
                    f"cached: ours=${our_pricing.cached:.3f} vs theirs=${their_cached:.3f}"
                )

        if errors:
            mismatches.append(f"{provider}/{model}: {'; '.join(errors)}")

    # Report findings
    if missing:
        print(f"\nModels not found in LiteLLM: {missing}")

    if mismatches:
        pytest.fail(
            f"Pricing mismatches found (tolerance={PRICE_TOLERANCE * 100}%):\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )


def test_grok_pricing_specifically(litellm_pricing):
    """Focused test on Grok pricing since it was just added."""
    grok_models = [
        ("grok-4", "xai/grok-4", 3.00, 15.00),
        ("grok-4-1-fast", "xai/grok-4-1-fast", 0.20, 0.50),
        ("grok-code-fast-1", "xai/grok-code-fast-1", 0.20, 1.50),
        ("grok-3-mini", "xai/grok-3-mini", 0.30, 0.50),
    ]

    for our_name, litellm_name, expected_input, expected_output in grok_models:
        assert our_name in GROK_PRICING, f"Missing {our_name} in GROK_PRICING"
        our = GROK_PRICING[our_name]

        # Check our values match expected
        assert our.input == expected_input, f"{our_name} input mismatch"
        assert our.output == expected_output, f"{our_name} output mismatch"

        # Check against LiteLLM if available
        if litellm_name in litellm_pricing:
            their = litellm_pricing[litellm_name]
            their_input = to_per_million(their.get("input_cost_per_token"))
            their_output = to_per_million(their.get("output_cost_per_token"))

            if their_input:
                assert prices_match(our.input, their_input), (
                    f"{our_name}: input ${our.input} != LiteLLM ${their_input:.2f}"
                )
            if their_output:
                assert prices_match(our.output, their_output), (
                    f"{our_name}: output ${our.output} != LiteLLM ${their_output:.2f}"
                )


# Map our Grok model names to pricepertoken's model_name field
GROK_TO_PRICEPERTOKEN = {
    "grok-4": "Grok 4",
    "grok-4-1-fast": "Grok 4.1 Fast",
    "grok-4-fast": "Grok 4 Fast",
    "grok-4-1-fast-non-reasoning": "Grok 4.1 Fast",  # same pricing as reasoning
    "grok-4-fast-non-reasoning": "Grok 4 Fast",  # same pricing as reasoning
    "grok-code-fast-1": "Grok Code Fast 1",
    "grok-3": "Grok 3",
    "grok-3-mini": "Grok 3 Mini",
}


@pytest.mark.unstable_external
def test_grok_pricing_against_pricepertoken(pricepertoken_xai):
    """Verify Grok pricing against pricepertoken.com.

    This provides a second source to cross-check against LiteLLM.
    If both sources disagree with us, we're likely wrong.
    If only one disagrees, one of the sources may be stale.
    """

    # Build lookup by model_name
    ppt_by_name = {m["model_name"]: m for m in pricepertoken_xai}

    mismatches = []
    for our_name, ppt_name in GROK_TO_PRICEPERTOKEN.items():
        if our_name not in GROK_PRICING:
            continue
        if ppt_name not in ppt_by_name:
            continue

        our = GROK_PRICING[our_name]
        their = ppt_by_name[ppt_name]

        their_input = their["pricing_prompt"]
        their_output = their["pricing_completion"]
        their_cached = their.get("pricing_input_cache_read")

        errors = []
        if not prices_match(our.input, their_input):
            errors.append(f"input: ours=${our.input:.2f} vs ppt=${their_input:.2f}")
        if not prices_match(our.output, their_output):
            errors.append(f"output: ours=${our.output:.2f} vs ppt=${their_output:.2f}")
        if their_cached and our.cached:
            if not prices_match(our.cached, their_cached):
                errors.append(
                    f"cached: ours=${our.cached:.3f} vs ppt=${their_cached:.3f}"
                )

        if errors:
            mismatches.append(f"{our_name}: {'; '.join(errors)}")

    if mismatches:
        pytest.fail(
            f"Pricing mismatches vs pricepertoken.com (tolerance={PRICE_TOLERANCE * 100}%):\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )


@pytest.mark.unstable_external
def test_sources_agree_on_grok(litellm_pricing, pricepertoken_xai):
    """Check that LiteLLM and pricepertoken.com agree on Grok pricing.

    If they disagree, one source may be stale. This helps identify which
    external source to trust when debugging pricing mismatches.
    """
    ppt_by_name = {m["model_name"]: m for m in pricepertoken_xai}

    # Models to cross-check (our_name, litellm_name, ppt_name)
    cross_check = [
        ("grok-4", "xai/grok-4", "Grok 4"),
        ("grok-4-1-fast", "xai/grok-4-1-fast", "Grok 4.1 Fast"),
        ("grok-code-fast-1", "xai/grok-code-fast-1", "Grok Code Fast 1"),
        ("grok-3-mini", "xai/grok-3-mini", "Grok 3 Mini"),
        ("grok-3", "xai/grok-3", "Grok 3"),
    ]

    disagreements = []
    for our_name, litellm_name, ppt_name in cross_check:
        if litellm_name not in litellm_pricing:
            continue
        if ppt_name not in ppt_by_name:
            continue

        ll = litellm_pricing[litellm_name]
        ll_input = to_per_million(ll.get("input_cost_per_token"))
        ll_output = to_per_million(ll.get("output_cost_per_token"))

        ppt = ppt_by_name[ppt_name]
        ppt_input = ppt["pricing_prompt"]
        ppt_output = ppt["pricing_completion"]

        errors = []
        if ll_input and not prices_match(ll_input, ppt_input):
            errors.append(f"input: litellm=${ll_input:.2f} vs ppt=${ppt_input:.2f}")
        if ll_output and not prices_match(ll_output, ppt_output):
            errors.append(f"output: litellm=${ll_output:.2f} vs ppt=${ppt_output:.2f}")

        if errors:
            disagreements.append(f"{our_name}: {'; '.join(errors)}")

    if disagreements:
        # Warning only - external sources may legitimately differ temporarily
        print("\nWARNING: External sources disagree on pricing:")
        for d in disagreements:
            print(f"  - {d}")
