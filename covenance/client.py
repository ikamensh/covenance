"""Instance-scoped LLM access with isolated keys and call records.

Routes to either native SDK clients or pydantic-ai backend based on provider.
Module-level helpers route through the default instance so legacy API keeps working.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ._caller_context import capture_caller_context, get_caller_info
from ._pydantic_ai_backend import ask_pydantic_ai
from ._routing import NATIVE_BACKEND_PROVIDERS, Backends, get_provider
from .exceptions import StructuredOutputParsingError
from .keys import (
    get_anthropic_api_key,
    get_gemini_api_key,
    get_grok_api_key,
    get_mistral_api_key,
    get_openai_api_key,
    get_openrouter_api_key,
)
from .record import Record, RecordStore, TokenUsage, get_env_records_dir
from .response_adapter import ResponseTypeAdapter

logger = logging.getLogger("covenance")

# Registry of all Covenance clients created in this process
_all_clients: list[Covenance] = []

# Re-export for backward compat with tests that import from client
_NATIVE_BACKEND_PROVIDERS = NATIVE_BACKEND_PROVIDERS
_get_provider = get_provider


def _normalize_model(model: str | object) -> str:
    """Convert model to string, handling enum values."""
    if hasattr(model, "value"):  # Enum
        return str(model.value)
    return str(model)


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
        self.backends = Backends()

        # Lazy-initialized native clients (only created if needed)
        self._native_clients: dict[str, Any] | None = None

        _all_clients.append(self)

    def get_record_store(self) -> RecordStore:
        return self._record_store

    def _get_key_overrides(self) -> dict[str, str | None]:
        """Return dict of API key overrides for backends."""
        return {
            "openai": self._openai_api_key,
            "anthropic": self._anthropic_api_key,
            "mistral": self._mistral_api_key,
            "gemini": self._gemini_api_key,
            "openrouter": self._openrouter_api_key,
            "grok": self._grok_api_key,
        }

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

    def _get_native_client(self, provider: str) -> Any | None:
        """Get or create native SDK client for provider."""
        from ._lazy_client import LazyClient
        from .keys import require_api_key

        if self._native_clients is None:
            self._native_clients = {}

        if provider in self._native_clients:
            return self._native_clients[provider]

        # Create client based on provider
        if provider == "openai":
            from openai import OpenAI

            client = LazyClient(
                lambda: OpenAI(
                    api_key=require_api_key(self._get_api_key("openai"), "openai")
                ),
                label="openai",
            )
        elif provider == "grok":
            from openai import OpenAI

            from .clients.grok_client import GROK_BASE_URL

            client = LazyClient(
                lambda: OpenAI(
                    api_key=require_api_key(self._get_api_key("grok"), "grok"),
                    base_url=GROK_BASE_URL,
                ),
                label="grok",
            )
        elif provider == "gemini":
            from google import genai

            client = LazyClient(
                lambda: genai.Client(
                    api_key=require_api_key(self._get_api_key("gemini"), "gemini")
                ),
                label="gemini",
            )
        elif provider == "mistral":
            from mistralai.client import Mistral as MistralClient

            client = LazyClient(
                lambda: MistralClient(
                    api_key=require_api_key(self._get_api_key("mistral"), "mistral")
                ),
                label="mistral",
            )
        elif provider == "anthropic":
            from anthropic import Anthropic

            client = LazyClient(
                lambda: Anthropic(
                    api_key=require_api_key(
                        self._get_api_key("anthropic"), "anthropic"
                    )
                ),
                label="anthropic",
            )
        elif provider == "openrouter":
            from openai import OpenAI

            from .clients.openrouter_client import OPENROUTER_BASE_URL

            client = LazyClient(
                lambda: OpenAI(
                    api_key=require_api_key(
                        self._get_api_key("openrouter"), "openrouter"
                    ),
                    base_url=OPENROUTER_BASE_URL,
                ),
                label="openrouter",
            )
        else:
            return None

        self._native_clients[provider] = client
        return client

    def _call_native_provider[T](
        self,
        user_msg: str,
        model: str,
        provider: str,
        llm_type: type[T] | None,
        sys_msg: str | None,
        temperature: float | None,
    ) -> Any:
        """Dispatch to the appropriate native SDK function. Returns BackendResult."""
        client = self._get_native_client(provider)

        if provider in ("openai", "grok"):
            from .clients.openai_client import ask_openai_compatible_structured

            return ask_openai_compatible_structured(
                client=client,
                user_msg=user_msg,
                response_type=llm_type,
                sys_msg=sys_msg,
                model=model,
                provider=provider,
                temperature=temperature,
            )
        elif provider == "gemini":
            from .clients.google_client import ask_gemini

            return ask_gemini(
                user_msg=user_msg,
                response_type=llm_type,
                sys_msg=sys_msg,
                model=model,
                client_override=client.resolve(),
                temperature=temperature,
            )
        elif provider == "mistral":
            from .clients.mistral_client import ask_mistral

            return ask_mistral(
                user_msg=user_msg,
                response_type=llm_type,
                sys_msg=sys_msg,
                model=model,
                client_override=client.resolve(),
                temperature=temperature,
            )
        elif provider == "anthropic":
            from .clients.anthropic_client import ask_anthropic

            return ask_anthropic(
                user_msg=user_msg,
                response_type=llm_type,
                sys_msg=sys_msg,
                model=model,
                client_override=client.resolve(),
                temperature=temperature,
            )
        elif provider == "openrouter":
            from .clients.openai_client import ask_openai_compatible_structured

            return ask_openai_compatible_structured(
                client=client,
                user_msg=user_msg,
                response_type=llm_type,
                sys_msg=sys_msg,
                model=model,
                provider="openrouter",
                temperature=temperature,
            )
        else:
            raise ValueError(f"No native backend for provider: {provider}")

    def _ask_native[T](
        self,
        user_msg: str,
        model: str,
        provider: str,
        response_type: type[T] | None,
        sys_msg: str | None,
        max_parsing_retries: int,
        temperature: float | None,
    ) -> Any:
        """Call native SDK backend. Returns BackendResult."""
        from ._backend_result import BackendResult

        # Adapt response_type for LLM API
        adapter = ResponseTypeAdapter(response_type)
        llm_type = adapter.get_llm_type()

        # Accumulate data across retries
        total_tpm_retries = 0
        total_tpm_wait = 0.0
        structured_output_retries = 0
        accumulated_input = 0
        accumulated_output = 0

        max_attempts = max_parsing_retries + 1

        for attempt in range(max_attempts):
            try:
                raw_result = self._call_native_provider(
                    user_msg=user_msg,
                    model=model,
                    provider=provider,
                    llm_type=llm_type,
                    sys_msg=sys_msg,
                    temperature=temperature,
                )

                # Accumulate TPM retry info
                total_tpm_retries += raw_result.tpm_retries
                total_tpm_wait += raw_result.tpm_wait_seconds

                result = raw_result.output

                if llm_type not in (None, str):
                    try:
                        result = TypeAdapter(llm_type).validate_python(result)
                    except ValidationError as exc:
                        raise StructuredOutputParsingError(
                            "Structured LLM output did not match expected schema.",
                            usage=raw_result.usage,
                        ) from exc

                # Calculate consolidated usage
                final_usage = raw_result.usage
                total_input = final_usage.prompt_tokens + accumulated_input
                total_output = final_usage.completion_tokens + accumulated_output

                consolidated_usage = TokenUsage(
                    prompt_tokens=total_input,
                    completion_tokens=total_output,
                    total_tokens=total_input + total_output,
                    cached_tokens=final_usage.cached_tokens,
                )

                return BackendResult(
                    output=adapter.unwrap(result),
                    usage=consolidated_usage,
                    tpm_retries=total_tpm_retries,
                    tpm_wait_seconds=total_tpm_wait,
                    structured_output_retries=structured_output_retries,
                )

            except StructuredOutputParsingError as e:
                if e.usage is not None:
                    accumulated_input += e.usage.prompt_tokens
                    accumulated_output += e.usage.completion_tokens
                structured_output_retries += 1
                if attempt == max_attempts - 1:
                    raise

        raise RuntimeError("_ask_native exhausted retry loop")

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
        """Route to appropriate backend and make LLM call.

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
        # Normalize model (handle enums)
        model = _normalize_model(model)
        provider = _get_provider(model)

        started_at = datetime.now(UTC)

        # Route to appropriate backend
        backend_name = self.backends.get(provider)
        logger.debug(
            "ask_llm: model=%s provider=%s backend=%s",
            model,
            provider,
            backend_name,
        )

        if backend_name == "native":
            backend_result = self._ask_native(
                user_msg=user_msg,
                model=model,
                provider=provider,
                response_type=response_type,
                sys_msg=sys_msg,
                max_parsing_retries=max_parsing_retries,
                temperature=temperature,
            )
        else:
            backend_result = ask_pydantic_ai(
                user_msg=user_msg,
                model=model,
                response_type=response_type,
                sys_msg=sys_msg,
                max_parsing_retries=max_parsing_retries,
                temperature=temperature,
                key_overrides=self._get_key_overrides(),
            )

        ended_at = datetime.now(UTC)

        # Record the call
        caller_function, caller_file, caller_line = get_caller_info()
        self._record_store.record_llm_call(
            model=model,
            provider=provider,
            backend=backend_name,
            usage=backend_result.usage,
            started_at=started_at,
            ended_at=ended_at,
            tpm_retries=backend_result.tpm_retries,
            tpm_retry_wait_seconds=backend_result.tpm_wait_seconds,
            structured_output_retries=backend_result.structured_output_retries,
            caller_function=caller_function,
            caller_file=caller_file,
            caller_line=caller_line,
        )

        return backend_result.output

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

    def print_call_timeline(self, width: int = 80) -> None:
        """Print a terminal-based waterfall chart of this client's LLM call timings."""
        from .visual import print_call_timeline as _print_call_timeline

        _print_call_timeline(records=self.get_records(), width=width)


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
