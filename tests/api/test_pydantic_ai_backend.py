"""Tests for pydantic-ai backend internals.

Doctest summary:
>>> retries = 0
>>> retries += 1
>>> retries
1
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

import covenance._pydantic_ai_backend as backend


@dataclass
class _NextAction:
    sleep: float


@dataclass
class _RetryState:
    next_action: _NextAction | None


def test_retry_tracker_accumulates_wait_and_retry_count():
    tracker = backend.RetryTracker()
    tracker.before_sleep(_RetryState(next_action=_NextAction(sleep=1.25)))
    tracker.before_sleep(_RetryState(next_action=None))
    tracker.after_retry(_RetryState(next_action=None))
    tracker.after_retry(_RetryState(next_action=None))

    assert tracker.wait_seconds == 1.25
    assert tracker.retries == 2


def test_create_retry_http_client_wires_transport_and_validation(monkeypatch):
    captured = {}

    class FakeTransport:
        def __init__(self, config, validate_response):
            captured["config"] = config
            captured["validate_response"] = validate_response

    class FakeAsyncClient:
        def __init__(self, transport, timeout):
            self.transport = transport
            self.timeout = timeout
            captured["transport"] = transport
            captured["timeout"] = timeout

    monkeypatch.setattr(backend, "AsyncTenacityTransport", FakeTransport)
    monkeypatch.setattr(backend.httpx, "AsyncClient", FakeAsyncClient)

    tracker = backend.RetryTracker()
    client = backend._create_retry_http_client(
        tracker=tracker, max_retries=7, max_wait=12.0, timeout=33.0
    )

    assert isinstance(client, FakeAsyncClient)
    assert captured["timeout"] == 33.0
    assert captured["transport"] is not None

    called = {"raised": False}

    class RetryableResponse:
        status_code = 429

        def raise_for_status(self):
            called["raised"] = True
            raise RuntimeError("raise_for_status called")

    class OkResponse:
        status_code = 200

        def raise_for_status(self):
            raise AssertionError("Should not raise for non-retryable status")

    with pytest.raises(RuntimeError, match="raise_for_status called"):
        captured["validate_response"](RetryableResponse())
    assert called["raised"] is True

    captured["validate_response"](OkResponse())


def _install_fake_provider_modules(monkeypatch):
    class OpenAIProvider:
        def __init__(self, api_key=None, http_client=None, base_url=None):
            self.api_key = api_key
            self.http_client = http_client
            self.base_url = base_url

    class OpenAIChatModel:
        def __init__(self, model_name, provider):
            self.model_name = model_name
            self.provider = provider

    class AsyncAnthropic:
        def __init__(self, api_key=None, http_client=None):
            self.api_key = api_key
            self.http_client = http_client

    class AnthropicProvider:
        def __init__(self, anthropic_client):
            self.anthropic_client = anthropic_client

    class AnthropicModel:
        def __init__(self, model_name, provider):
            self.model_name = model_name
            self.provider = provider

    class GoogleProvider:
        def __init__(self, api_key=None, http_client=None):
            self.api_key = api_key
            self.http_client = http_client

    class GoogleModel:
        def __init__(self, model_name, provider):
            self.model_name = model_name
            self.provider = provider

    class MistralClient:
        def __init__(self, api_key=None, async_client=None):
            self.api_key = api_key
            self.async_client = async_client

    class MistralProvider:
        def __init__(self, mistral_client):
            self.mistral_client = mistral_client

    class MistralModel:
        def __init__(self, model_name, provider):
            self.model_name = model_name
            self.provider = provider

    monkeypatch.setitem(sys.modules, "pydantic_ai.models.openai", types.ModuleType("x"))
    monkeypatch.setitem(
        sys.modules, "pydantic_ai.providers.openai", types.ModuleType("x")
    )
    monkeypatch.setitem(
        sys.modules, "pydantic_ai.models.anthropic", types.ModuleType("x")
    )
    monkeypatch.setitem(
        sys.modules, "pydantic_ai.providers.anthropic", types.ModuleType("x")
    )
    monkeypatch.setitem(sys.modules, "pydantic_ai.models.google", types.ModuleType("x"))
    monkeypatch.setitem(
        sys.modules, "pydantic_ai.providers.google", types.ModuleType("x")
    )
    monkeypatch.setitem(
        sys.modules, "pydantic_ai.models.mistral", types.ModuleType("x")
    )
    monkeypatch.setitem(
        sys.modules, "pydantic_ai.providers.mistral", types.ModuleType("x")
    )
    monkeypatch.setitem(sys.modules, "anthropic", types.ModuleType("x"))
    monkeypatch.setitem(sys.modules, "mistralai", types.ModuleType("x"))

    sys.modules["pydantic_ai.models.openai"].OpenAIChatModel = OpenAIChatModel
    sys.modules["pydantic_ai.providers.openai"].OpenAIProvider = OpenAIProvider
    sys.modules["anthropic"].AsyncAnthropic = AsyncAnthropic
    sys.modules["pydantic_ai.models.anthropic"].AnthropicModel = AnthropicModel
    sys.modules["pydantic_ai.providers.anthropic"].AnthropicProvider = (
        AnthropicProvider
    )
    sys.modules["pydantic_ai.models.google"].GoogleModel = GoogleModel
    sys.modules["pydantic_ai.providers.google"].GoogleProvider = GoogleProvider
    sys.modules["mistralai"].Mistral = MistralClient
    sys.modules["pydantic_ai.models.mistral"].MistralModel = MistralModel
    sys.modules["pydantic_ai.providers.mistral"].MistralProvider = MistralProvider


def test_create_model_with_retry_supports_all_providers(monkeypatch):
    _install_fake_provider_modules(monkeypatch)
    monkeypatch.setattr(backend, "_create_retry_http_client", lambda tracker: "HTTP")

    tracker = backend.RetryTracker()
    key_overrides = {
        "openai": "k-openai",
        "anthropic": "k-anthropic",
        "gemini": "k-gemini",
        "mistral": "k-mistral",
        "openrouter": "k-openrouter",
        "grok": "k-grok",
    }

    openai_model = backend._create_model_with_retry(
        "gpt-4.1-nano", "openai", tracker, key_overrides
    )
    assert openai_model.provider.api_key == "k-openai"

    anthropic_model = backend._create_model_with_retry(
        "claude-haiku", "anthropic", tracker, key_overrides
    )
    assert anthropic_model.provider.anthropic_client.api_key == "k-anthropic"

    gemini_model = backend._create_model_with_retry(
        "gemini-2.5-flash", "gemini", tracker, key_overrides
    )
    assert gemini_model.provider.api_key == "k-gemini"

    mistral_model = backend._create_model_with_retry(
        "mistral-small", "mistral", tracker, key_overrides
    )
    assert mistral_model.provider.mistral_client.api_key == "k-mistral"

    openrouter_model = backend._create_model_with_retry(
        "openai/gpt-4o-mini", "openrouter", tracker, key_overrides
    )
    assert openrouter_model.provider.api_key == "k-openrouter"
    assert openrouter_model.provider.base_url == "https://openrouter.ai/api/v1"

    grok_model = backend._create_model_with_retry(
        "grok-3-mini", "grok", tracker, key_overrides
    )
    assert grok_model.provider.api_key == "k-grok"
    assert grok_model.provider.base_url == "https://api.x.ai/v1"


def test_create_model_with_retry_uses_xai_env_fallback(monkeypatch):
    _install_fake_provider_modules(monkeypatch)
    monkeypatch.setattr(backend, "_create_retry_http_client", lambda tracker: "HTTP")
    monkeypatch.setattr(backend, "_get_api_key", lambda provider, key_overrides: None)
    monkeypatch.setenv("XAI_API_KEY", "xai-from-env")

    model = backend._create_model_with_retry(
        "grok-3-mini", "grok", backend.RetryTracker(), {}
    )
    assert model.provider.api_key == "xai-from-env"


def test_create_model_with_retry_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(backend, "_create_retry_http_client", lambda tracker: "HTTP")
    with pytest.raises(ValueError, match="Unknown provider"):
        backend._create_model_with_retry("some-model", "unknown", backend.RetryTracker(), {})


def test_get_api_key_prefers_override_and_falls_back_to_getters(monkeypatch):
    monkeypatch.setattr(backend, "get_openai_api_key", lambda: "from-env")
    override = backend._get_api_key("openai", {"openai": "from-override"})
    fallback = backend._get_api_key("openai", {})
    unknown = backend._get_api_key("unknown", {})

    assert override == "from-override"
    assert fallback == "from-env"
    assert unknown is None


def test_ask_pydantic_ai_plain_text_path(monkeypatch):
    class FakeAdapter:
        def __init__(self, response_type):
            self.response_type = response_type

        def get_llm_type(self):
            return None

        def unwrap(self, value):
            return value

    class FakeUsage:
        input_tokens = 10
        output_tokens = 7
        total_tokens = 17
        cache_read_tokens = 2

    class FakeResult:
        def __init__(self):
            self.output = "plain text output"

        def usage(self):
            return FakeUsage()

    captured = {}

    class FakeAgent:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

        def run_sync(self, user_msg, model_settings):
            captured["user_msg"] = user_msg
            captured["model_settings"] = model_settings
            return FakeResult()

    monkeypatch.setattr(backend, "ResponseTypeAdapter", FakeAdapter)
    monkeypatch.setattr(backend, "Agent", FakeAgent)
    monkeypatch.setattr(backend, "get_provider", lambda model: "openai")
    monkeypatch.setattr(backend, "_create_model_with_retry", lambda *args, **kwargs: "M")

    result = backend.ask_pydantic_ai(
        user_msg="hello",
        model="gpt-4o-mini",
        response_type=None,
        sys_msg="be concise",
        temperature=0.3,
    )

    assert result.output == "plain text output"
    assert result.usage.total_tokens == 17
    assert result.structured_output_retries == 0
    assert "output_type" not in captured["kwargs"]
    assert captured["model_settings"] == {"temperature": 0.3}


def test_ask_pydantic_ai_structured_path_unwraps_and_reads_retries(monkeypatch):
    class FakeAdapter:
        def __init__(self, response_type):
            self.response_type = response_type

        def get_llm_type(self):
            return "LLM_TYPE"

        def unwrap(self, value):
            return {"unwrapped": value}

    class FakeUsage:
        input_tokens = 20
        output_tokens = 5
        total_tokens = 25
        cache_read_tokens = 0

    class FakeState:
        retries = 4

    class FakeResult:
        def __init__(self):
            self.output = {"raw": "value"}
            self._state = FakeState()

        def usage(self):
            return FakeUsage()

    captured = {}

    class FakeAgent:
        def __init__(self, model, **kwargs):
            captured["kwargs"] = kwargs

        def run_sync(self, user_msg, model_settings):
            return FakeResult()

    monkeypatch.setattr(backend, "ResponseTypeAdapter", FakeAdapter)
    monkeypatch.setattr(backend, "Agent", FakeAgent)
    monkeypatch.setattr(backend, "get_provider", lambda model: "mistral")
    monkeypatch.setattr(backend, "_create_model_with_retry", lambda *args, **kwargs: "M")

    result = backend.ask_pydantic_ai(
        user_msg="hello",
        model="mistral-small",
        response_type=int,
        max_parsing_retries=3,
    )

    assert result.output == {"unwrapped": {"raw": "value"}}
    assert result.structured_output_retries == 4
    assert captured["kwargs"]["output_type"] == "LLM_TYPE"
    assert captured["kwargs"]["retries"] == 3
