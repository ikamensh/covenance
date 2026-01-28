"""Fetch and display pricing from LiteLLM and PricePerToken.

Usage: python scripts/fetch_pricing.py

Useful for updating pricing data when adding new models.
"""

import json
import urllib.request

# --- Configuration ---

LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

PRICEPERTOKEN_MCP_URL = "https://api.pricepertoken.com/mcp/mcp"

# Providers to fetch from PricePerToken
PRICEPERTOKEN_AUTHORS = ["openai", "google", "xai", "anthropic", "mistralai"]

# LiteLLM model prefixes to filter (set to None for all)
LITELLM_PREFIXES = ["gpt-", "o1", "o3", "o4", "gemini-", "xai/", "claude-", "mistral"]


# --- Fetch functions ---


def fetch_litellm_pricing() -> dict:
    """Fetch and parse LiteLLM pricing JSON."""
    print("Fetching LiteLLM pricing...")
    with urllib.request.urlopen(LITELLM_PRICING_URL, timeout=30) as response:
        return json.loads(response.read().decode())


def fetch_pricepertoken_models(author: str) -> list[dict]:
    """Fetch models from pricepertoken.com MCP API."""
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
            "User-Agent": "covenance-pricing-fetch/1.0",
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        resp = json.loads(response.read().decode())
        return json.loads(resp["result"]["content"][0]["text"])


def to_per_million(cost_per_token: float | None) -> float | None:
    """Convert cost per token to cost per million tokens."""
    if cost_per_token is None:
        return None
    return round(cost_per_token * 1_000_000, 4)


# --- Display functions ---


def display_litellm_pricing(data: dict):
    """Display filtered LiteLLM pricing."""
    print("\n" + "=" * 80)
    print("LITELLM PRICING (per 1M tokens)")
    print("=" * 80)

    filtered = {}
    for model, info in data.items():
        if LITELLM_PREFIXES:
            if not any(
                model.startswith(p) or model.startswith(p.replace("/", "_"))
                for p in LITELLM_PREFIXES
            ):
                continue
        filtered[model] = info

    # Group by prefix
    groups = {}
    for model in sorted(filtered.keys()):
        prefix = model.split("-")[0].split("/")[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(model)

    for prefix in sorted(groups.keys()):
        print(f"\n--- {prefix.upper()} ---")
        for model in groups[prefix]:
            info = filtered[model]
            inp = to_per_million(info.get("input_cost_per_token"))
            out = to_per_million(info.get("output_cost_per_token"))
            cached = to_per_million(info.get("cache_read_input_token_cost"))

            inp_str = f"${inp:.4f}" if inp else "N/A"
            out_str = f"${out:.4f}" if out else "N/A"
            cached_str = f"${cached:.4f}" if cached else "N/A"

            print(f"  {model:45} in={inp_str:12} out={out_str:12} cached={cached_str}")


def display_pricepertoken_pricing(author: str, data: list[dict]):
    """Display PricePerToken pricing for an author."""
    print(f"\n--- {author.upper()} (PricePerToken) ---")

    for model in sorted(data, key=lambda m: m.get("model_name", "")):
        name = model.get("model_name", "?")
        model_id = model.get("model_id", "?")
        inp = model.get("pricing_prompt")
        out = model.get("pricing_completion")
        cached = model.get("pricing_input_cache_read")

        inp_str = f"${inp:.4f}" if inp else "N/A"
        out_str = f"${out:.4f}" if out else "N/A"
        cached_str = f"${cached:.4f}" if cached else "N/A"

        print(
            f"  {name:30} ({model_id:35}) in={inp_str:12} out={out_str:12} cached={cached_str}"
        )


# --- Main ---

if __name__ == "__main__":
    # Fetch and display LiteLLM
    litellm_data = fetch_litellm_pricing()
    display_litellm_pricing(litellm_data)

    # Fetch and display PricePerToken
    print("\n" + "=" * 80)
    print("PRICEPERTOKEN PRICING (per 1M tokens)")
    print("=" * 80)

    for author in PRICEPERTOKEN_AUTHORS:
        try:
            print(f"\nFetching {author}...")
            ppt_data = fetch_pricepertoken_models(author)
            display_pricepertoken_pricing(author, ppt_data)
        except Exception as e:
            print(f"  Error fetching {author}: {e}")

    print("\n" + "=" * 80)
    print("Done.")
