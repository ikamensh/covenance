"""Additional tests for API key helpers.

Doctest summary:
>>> "OPENAI_API_KEY" in ["OPENAI_API_KEY", "GEMINI_API_KEY"]
True
"""

import pytest

import covenance.keys as keys


def test_first_env_returns_none_when_no_candidates(monkeypatch):
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    assert keys._first_env("A", "B") is None


@pytest.mark.parametrize(
    ("env_name", "getter_name"),
    [
        ("ANTHROPIC_API_KEY", "get_anthropic_api_key"),
        ("MISTRAL_API_KEY", "get_mistral_api_key"),
        ("OPENROUTER_API_KEY", "get_openrouter_api_key"),
        ("XAI_API_KEY", "get_grok_api_key"),
    ],
)
def test_provider_specific_getters_read_env(monkeypatch, env_name, getter_name):
    monkeypatch.setattr(keys, "load_env_if_present", lambda: None)
    if getter_name == "get_grok_api_key":
        monkeypatch.delenv("GROK_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv(env_name, "secret-value")
    getter = getattr(keys, getter_name)
    assert getter() == "secret-value"


def test_require_api_key_uses_explicit_env_var_names_in_error():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY or ALT_KEY"):
        keys.require_api_key(None, "openai", env_vars=["OPENAI_API_KEY", "ALT_KEY"])


def test_require_api_key_with_unknown_provider_still_raises_clear_error():
    with pytest.raises(RuntimeError, match="Missing unknown API key"):
        keys.require_api_key(None, "unknown")


def test_get_grok_api_key_prefers_grok_name_when_both_are_set(monkeypatch):
    monkeypatch.setattr(keys, "load_env_if_present", lambda: None)
    monkeypatch.setenv("GROK_API_KEY", "grok-value")
    monkeypatch.setenv("XAI_API_KEY", "xai-value")
    assert keys.get_grok_api_key() == "grok-value"


def test_get_grok_api_key_falls_back_to_xai_name(monkeypatch):
    monkeypatch.setattr(keys, "load_env_if_present", lambda: None)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-value")
    assert keys.get_grok_api_key() == "xai-value"
