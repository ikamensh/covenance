"""Pydantic-AI backend for LLM calls.

Uses pydantic-ai as the unified backend for multi-provider LLM calls.
This backend is the default for most providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from httpx import HTTPStatusError
from pydantic_ai import Agent
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import RetryCallState, retry_if_exception_type, stop_after_attempt

from ._backend_result import BackendResult
from ._routing import get_provider
from .keys import (
    get_anthropic_api_key,
    get_gemini_api_key,
    get_grok_api_key,
    get_mistral_api_key,
    get_openai_api_key,
    get_openrouter_api_key,
)
from .record import TokenUsage
from .response_adapter import ResponseTypeAdapter


@dataclass
class RetryTracker:
    """Tracks retry attempts and wait times for a single LLM call."""

    retries: int = 0
    wait_seconds: float = 0.0

    def before_sleep(self, retry_state: RetryCallState) -> None:
        """Called before sleeping between retries."""
        wait_time = retry_state.next_action.sleep if retry_state.next_action else 0.0
        self.wait_seconds += wait_time

    def after_retry(self, retry_state: RetryCallState) -> None:
        """Called after each retry attempt."""
        self.retries += 1


def _create_retry_http_client(
    tracker: RetryTracker,
    max_retries: int = 10,
    max_wait: float = 300.0,
    timeout: float = 600.0,
) -> httpx.AsyncClient:
    """Create an HTTP client with rate limit retry support."""

    def validate_response(response: httpx.Response) -> None:
        """Raise for retryable status codes (429 rate limit, 5xx server errors)."""
        if response.status_code in (429, 502, 503, 504):
            response.raise_for_status()

    transport = AsyncTenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception_type(HTTPStatusError),
            wait=wait_retry_after(max_wait=max_wait),
            stop=stop_after_attempt(max_retries),
            before_sleep=tracker.before_sleep,
            after=tracker.after_retry,
            reraise=True,
        ),
        validate_response=validate_response,
    )
    return httpx.AsyncClient(transport=transport, timeout=timeout)


def _get_api_key(provider: str, key_overrides: dict[str, str | None]) -> str | None:
    """Get API key for provider (explicit override or from env)."""
    key_getters = {
        "openai": get_openai_api_key,
        "anthropic": get_anthropic_api_key,
        "mistral": get_mistral_api_key,
        "gemini": get_gemini_api_key,
        "openrouter": get_openrouter_api_key,
        "grok": get_grok_api_key,
    }
    override = key_overrides.get(provider)
    if override:
        return override
    getter = key_getters.get(provider)
    return getter() if getter else None


def _create_model_with_retry(
    model_name: str,
    provider: str,
    tracker: RetryTracker,
    key_overrides: dict[str, str | None],
) -> Any:
    """Create a pydantic-ai model with retry-enabled HTTP client."""
    api_key = _get_api_key(provider, key_overrides)
    http_client = _create_retry_http_client(tracker)

    if provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(api_key=api_key, http_client=http_client),
        )

    elif provider == "anthropic":
        from anthropic import AsyncAnthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        anthropic_client = AsyncAnthropic(api_key=api_key, http_client=http_client)
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(anthropic_client=anthropic_client),
        )

    elif provider == "gemini":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(
            model_name,
            provider=GoogleProvider(api_key=api_key, http_client=http_client),
        )

    elif provider == "mistral":
        from mistralai import Mistral
        from pydantic_ai.models.mistral import MistralModel
        from pydantic_ai.providers.mistral import MistralProvider

        mistral_client = Mistral(api_key=api_key, async_client=http_client)
        return MistralModel(
            model_name, provider=MistralProvider(mistral_client=mistral_client)
        )

    elif provider == "grok":
        # xAI uses OpenAI-compatible API
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        grok_key = api_key or os.environ.get("XAI_API_KEY")
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(
                api_key=grok_key,
                base_url="https://api.x.ai/v1",
                http_client=http_client,
            ),
        )

    elif provider == "openrouter":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                http_client=http_client,
            ),
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def ask_pydantic_ai[T](
    user_msg: str,
    model: str,
    response_type: type[T] | None = None,
    sys_msg: str | None = None,
    *,
    max_parsing_retries: int = 2,
    temperature: float | None = None,
    key_overrides: dict[str, str | None] | None = None,
) -> BackendResult:
    """Make LLM call using pydantic-ai backend. Returns BackendResult."""
    provider = get_provider(model)
    key_overrides = key_overrides or {}

    # Create retry tracker and model with retry-enabled HTTP client
    tracker = RetryTracker()
    pydantic_model = _create_model_with_retry(model, provider, tracker, key_overrides)

    # Adapt response_type for LLM API (wrap if needed)
    adapter = ResponseTypeAdapter(response_type)
    llm_type = adapter.get_llm_type()

    # Determine output_type for pydantic-ai
    is_plain_text = llm_type is None or llm_type is str

    # Build model settings
    model_settings: dict[str, Any] = {}
    if temperature is not None:
        model_settings["temperature"] = temperature

    # Create agent - don't pass output_type if plain text
    agent_kwargs: dict[str, Any] = {
        "instructions": sys_msg,
        "retries": max_parsing_retries,
    }
    if not is_plain_text:
        agent_kwargs["output_type"] = llm_type

    agent = Agent(pydantic_model, **agent_kwargs)
    result = agent.run_sync(user_msg, model_settings=model_settings)

    # Extract usage from pydantic-ai result
    usage = result.usage()
    token_usage = TokenUsage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_tokens=usage.cache_read_tokens,
    )

    # Extract structured output retries from pydantic-ai internal state
    structured_retries = getattr(result, "_state", None)
    structured_retries = (
        getattr(structured_retries, "retries", 0) if structured_retries else 0
    )

    # Unwrap result if needed
    output = adapter.unwrap(result.output) if not is_plain_text else result.output

    return BackendResult(
        output=output,
        usage=token_usage,
        tpm_retries=tracker.retries,
        tpm_wait_seconds=tracker.wait_seconds,
        structured_output_retries=structured_retries,
    )
