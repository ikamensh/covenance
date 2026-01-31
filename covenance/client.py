"""Instance-scoped LLM access with isolated keys and call records.

Uses pydantic-ai as the backend for multi-provider LLM calls.
Module-level helpers route through the default instance so legacy API keeps working.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from httpx import HTTPStatusError
from pydantic_ai import Agent
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import RetryCallState, retry_if_exception_type, stop_after_attempt

from ._caller_context import capture_caller_context, get_caller_info
from .keys import (
    get_anthropic_api_key,
    get_gemini_api_key,
    get_grok_api_key,
    get_mistral_api_key,
    get_openai_api_key,
    get_openrouter_api_key,
)
from .record import Record, RecordStore, TokenUsage, get_env_records_dir


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
    """Create an HTTP client with rate limit retry support.

    Args:
        tracker: RetryTracker to record retry stats
        max_retries: Maximum number of retry attempts (default: 10)
        max_wait: Maximum wait time per retry in seconds (default: 300s / 5min)
        timeout: HTTP request timeout in seconds (default: 600s / 10min)
    """

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


# Registry of all Covenance clients created in this process
_all_clients: list[Covenance] = []


def _normalize_model(model: str | object) -> str:
    """Convert model to string, handling enum values."""
    if hasattr(model, "value"):  # Enum
        return str(model.value)
    return str(model)


def _get_pydantic_ai_model(model: str, provider: str) -> str:
    """Convert covenance model name to pydantic-ai model spec.

    pydantic-ai uses format 'provider:model_name', e.g.:
    - openai:gpt-4
    - anthropic:claude-3-opus
    - google-gla:gemini-2.5-flash
    - mistral:mistral-small-latest
    - xai:grok-2
    - openrouter:provider/model
    """
    provider_map = {
        "openai": "openai",
        "anthropic": "anthropic",
        "gemini": "google-gla",
        "mistral": "mistral",
        "grok": "xai",
        "openrouter": "openrouter",
    }
    pydantic_provider = provider_map.get(provider, provider)
    return f"{pydantic_provider}:{model}"


def _get_provider(model: str) -> str:
    """Determine provider from model name."""
    if model.startswith("gemini"):
        return "gemini"
    elif model.startswith(("mistral", "ministral", "codestral")):
        return "mistral"
    elif model.startswith("claude"):
        return "anthropic"
    elif model.startswith("grok"):
        return "grok"
    elif "/" in model:
        return "openrouter"
    else:
        return "openai"


class Covenance:
    """LLM client with isolated API keys and call records.

    Each instance maintains its own record store and can have its own API keys.
    This allows multiple independent LLM configurations in the same process.
    """

    def __init__(
        self,
        *,
        label: str | None = None,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        mistral_api_key: str | None = None,
        gemini_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        grok_api_key: str | None = None,
        records_dir: str | Path | None = None,
    ) -> None:
        self.label = label
        self._openai_api_key = openai_api_key
        self._anthropic_api_key = anthropic_api_key
        self._mistral_api_key = mistral_api_key
        self._gemini_api_key = gemini_api_key
        self._openrouter_api_key = openrouter_api_key
        self._grok_api_key = grok_api_key
        self._record_store = RecordStore(records_dir=records_dir, label=label)
        _all_clients.append(self)

    def get_record_store(self) -> RecordStore:
        return self._record_store

    def _get_api_key(self, provider: str) -> str | None:
        """Get API key for provider (explicit override or from env)."""
        key_getters: dict[str, tuple[str | None, Callable[[], str | None]]] = {
            "openai": (self._openai_api_key, get_openai_api_key),
            "anthropic": (self._anthropic_api_key, get_anthropic_api_key),
            "mistral": (self._mistral_api_key, get_mistral_api_key),
            "gemini": (self._gemini_api_key, get_gemini_api_key),
            "openrouter": (self._openrouter_api_key, get_openrouter_api_key),
            "grok": (self._grok_api_key, get_grok_api_key),
        }
        if provider in key_getters:
            override, getter = key_getters[provider]
            return override or getter()
        return None

    def _create_model_with_retry(
        self, model_name: str, provider: str, tracker: RetryTracker
    ) -> Any:
        """Create a pydantic-ai model with retry-enabled HTTP client.

        Returns a model object suitable for Agent(), with automatic rate limit retries.
        """
        api_key = self._get_api_key(provider)
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
            # xAI native SDK doesn't support custom http_client, use OpenAI-compatible
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

    def ask_llm[T](
        self,
        user_msg: str,
        model: str,
        response_type: type[T] | None = None,
        sys_msg: str | None = None,
        *,
        max_parsing_retries: int = 2,
        temperature: float | None = None,
    ) -> T:
        """Route to pydantic-ai Agent and make LLM call.

        Args:
            user_msg: User message/prompt
            model: Model name - determines provider routing
            response_type: Type for structured output. Can be:
                - None or str: returns plain text
                - Pydantic model: returns model instance
                - int, bool, float, list[X], tuple[...] - simple python types
            sys_msg: Optional system message
            max_parsing_retries: Retries for structured output parsing errors
            temperature: Sampling temperature
        """
        from .response_adapter import ResponseTypeAdapter

        # Normalize model (handle enums)
        model = _normalize_model(model)
        provider = _get_provider(model)

        # Create retry tracker and model with retry-enabled HTTP client
        tracker = RetryTracker()
        pydantic_model = self._create_model_with_retry(model, provider, tracker)

        # Adapt response_type for LLM API (wrap if needed)
        adapter = ResponseTypeAdapter(response_type)
        llm_type = adapter.get_llm_type()

        started_at = datetime.now(UTC)

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

        ended_at = datetime.now(UTC)

        # Extract usage from pydantic-ai result
        usage = result.usage()
        token_usage = TokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=usage.cache_read_tokens,
        )

        # Extract structured output retries from pydantic-ai internal state
        # Note: _state is private but the only way to access retry count
        structured_retries = getattr(result, "_state", None)
        structured_retries = (
            getattr(structured_retries, "retries", 0) if structured_retries else 0
        )

        # Record the call with retry stats
        caller_function, caller_file, caller_line = get_caller_info()
        self._record_store.record_llm_call(
            model=model,
            provider=provider,
            usage=token_usage,
            started_at=started_at,
            ended_at=ended_at,
            tpm_retries=tracker.retries,
            tpm_retry_wait_seconds=tracker.wait_seconds,
            structured_output_retries=structured_retries,
            caller_function=caller_function,
            caller_file=caller_file,
            caller_line=caller_line,
        )

        # Unwrap result if needed
        return adapter.unwrap(result.output) if not is_plain_text else result.output

    def llm_consensus[T](
        self,
        user_msg: str,
        model: str,
        response_type: type[T] | None = None,
        sys_msg: str | None = None,
        *,
        num_candidates: int = 3,
        additional_models: list[str] | None = None,
        integration_model: str | None = None,
        parallel: bool = True,
    ) -> T:
        """Make multiple LLM calls and integrate results.

        Args:
            user_msg: User message/prompt
            model: Model name for candidate generation
            response_type: Type for structured output
            sys_msg: Optional system message
            num_candidates: Number of parallel calls (default: 3)
            additional_models: Models to cycle through for workers
            integration_model: Model for integration (defaults to same as model)
            parallel: Whether to make calls in parallel (default: True)
        """
        capture_caller_context()

        if num_candidates == 1:
            return self.ask_llm(
                user_msg=user_msg,
                response_type=response_type,
                sys_msg=sys_msg,
                model=model,
            )

        if integration_model is None:
            integration_model = model

        worker_models = additional_models if additional_models else [model]

        def make_candidate_call(call_index: int) -> T:
            worker_model = worker_models[call_index % len(worker_models)]
            return self.ask_llm(
                user_msg=user_msg,
                response_type=response_type,
                sys_msg=sys_msg,
                model=worker_model,
            )

        candidates: list[T] = []
        if parallel:
            with ThreadPoolExecutor(max_workers=num_candidates) as executor:
                futures = [
                    executor.submit(copy_context().run, make_candidate_call, i)
                    for i in range(num_candidates)
                ]
                for future in as_completed(futures):
                    candidates.append(future.result())
        else:
            for i in range(num_candidates):
                candidates.append(make_candidate_call(i))

        # Format candidates for integration
        candidate_texts = []
        for i, candidate in enumerate(candidates, 1):
            if hasattr(candidate, "model_dump"):
                candidate_json = json.dumps(
                    candidate.model_dump(mode="json"), ensure_ascii=False, indent=2
                )
            elif isinstance(candidate, (dict, list)):
                candidate_json = json.dumps(candidate, ensure_ascii=False, indent=2)
            else:
                candidate_json = str(candidate)
            candidate_texts.append(f"--- Candidate Answer {i} ---\n{candidate_json}")

        integration_user_msg = f"""{user_msg}

Below are {len(candidates)} candidate answers generated by worker LLMs. Please integrate them into a single, high-quality answer that follows the same format and requirements as specified above.

"""
        for candidate_text in candidate_texts:
            integration_user_msg += f"\n{candidate_text}\n"

        integration_sys_msg = (
            "You are an LLM orchestrator, your goal is to integrate individual answers into a high quality answer. "
            f"Worker system message: {sys_msg or 'you are a helpful assistant'}"
        )

        return self.ask_llm(
            user_msg=integration_user_msg,
            response_type=response_type,
            sys_msg=integration_sys_msg,
            model=integration_model,
        )

    def get_records(self) -> list[Record]:
        return self._record_store.get_records()

    def clear_records(self) -> None:
        self._record_store.clear_records()

    def usage_summary(self) -> dict:
        """Compute usage summary from this client's records."""
        from .record import usage_summary as _usage_summary

        return _usage_summary(records=self.get_records())

    def print_usage(self, title: str | None = None, cost_format: str = "plain") -> None:
        """Print a formatted usage summary to stdout for this client's records."""
        from .record import print_usage as _print_usage

        if title is None:
            label = self.label or "default client"
            title = f"LLM Usage Summary ({label})"
        _print_usage(records=self.get_records(), title=title, cost_format=cost_format)


_default_client = Covenance(label="default client", records_dir=get_env_records_dir())


def get_all_clients() -> list[Covenance]:
    """Return list of all Covenance clients created in this process."""
    return _all_clients.copy()


def get_all_records() -> list[Record]:
    """Return all records from all Covenance clients, sorted by start time."""
    all_records = []
    for client in _all_clients:
        all_records.extend(client.get_records())
    all_records.sort(key=lambda r: r.started_at)
    return all_records


def ask_llm[T](
    user_msg: str,
    model: str,
    response_type: type[T] | None = None,
    sys_msg: str | None = None,
    *,
    max_parsing_retries: int = 2,
    temperature: float | None = None,
) -> T:
    """See docstring in the class method."""
    return _default_client.ask_llm(
        user_msg=user_msg,
        model=model,
        response_type=response_type,
        sys_msg=sys_msg,
        max_parsing_retries=max_parsing_retries,
        temperature=temperature,
    )


def llm_consensus[T](
    user_msg: str,
    model: str,
    response_type: type[T] | None = None,
    sys_msg: str | None = None,
    *,
    num_candidates: int = 3,
    additional_models: list[str] | None = None,
    integration_model: str | None = None,
    parallel: bool = True,
) -> T:
    """See docstring in the class method."""
    return _default_client.llm_consensus(
        user_msg=user_msg,
        model=model,
        response_type=response_type,
        sys_msg=sys_msg,
        num_candidates=num_candidates,
        additional_models=additional_models,
        integration_model=integration_model,
        parallel=parallel,
    )
