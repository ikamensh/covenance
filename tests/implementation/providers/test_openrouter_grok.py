"""Tests for OpenRouter/Grok wrapper clients.

Doctest summary:
>>> provider = "openrouter"
>>> provider.upper()
'OPENROUTER'
"""

import openai

from covenance.clients import grok_client, openrouter_client


def test_create_openrouter_client_uses_expected_base_url(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(openrouter_client, "require_provider", lambda name: None)
    monkeypatch.setattr(openrouter_client, "get_openrouter_api_key", lambda: "or-key")
    monkeypatch.setattr(openrouter_client, "require_api_key", lambda key, provider: key)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    client = openrouter_client._create_openrouter_client()
    assert isinstance(client, FakeOpenAI)
    assert captured["api_key"] == "or-key"
    assert captured["base_url"] == openrouter_client.OPENROUTER_BASE_URL


def test_ask_openrouter_forwards_to_openai_compatible_helper(monkeypatch):
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(openrouter_client, "ask_openai_compatible_structured", fake_call)

    result = openrouter_client.ask_openrouter(
        "hello",
        response_type=dict,
        sys_msg="s",
        model="meta-llama/llama-3.1-8b",
        client_override="CLIENT",
        temperature=0.2,
    )
    assert result == "ok"
    assert captured["provider"] == "openrouter"
    assert captured["client"] == "CLIENT"
    assert captured["temperature"] == 0.2


def test_create_grok_client_uses_expected_base_url(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(grok_client, "require_provider", lambda name: None)
    monkeypatch.setattr(grok_client, "get_grok_api_key", lambda: "grok-key")
    monkeypatch.setattr(grok_client, "require_api_key", lambda key, provider: key)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    client = grok_client._create_grok_client()
    assert isinstance(client, FakeOpenAI)
    assert captured["api_key"] == "grok-key"
    assert captured["base_url"] == grok_client.GROK_BASE_URL


def test_ask_grok_forwards_to_openai_compatible_helper(monkeypatch):
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(grok_client, "ask_openai_compatible_structured", fake_call)

    result = grok_client.ask_grok(
        "hello",
        response_type=int,
        sys_msg="s",
        model="grok-4-fast",
        client_override="CLIENT",
        temperature=0.8,
    )
    assert result == "ok"
    assert captured["provider"] == "grok"
    assert captured["client"] == "CLIENT"
    assert captured["temperature"] == 0.8
