"""Instance-scoped LLM access with isolated keys and call records.

Module-level helpers route through the default instance so legacy API keeps working.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from google import genai
from mistralai import Mistral
from openai import OpenAI
from pydantic import TypeAdapter, ValidationError

from ._lazy_client import LazyClient
from .clients.grok_client import GROK_BASE_URL
from .clients.openrouter_client import OPENROUTER_BASE_URL
from .exceptions import StructuredOutputParsingError
from .keys import (
    get_anthropic_api_key,
    get_gemini_api_key,
    get_grok_api_key,
    get_mistral_api_key,
    get_openai_api_key,
    get_openrouter_api_key,
    require_api_key,
)
from .record import Record, RecordStore, get_env_records_dir
from .response_adapter import ResponseTypeAdapter


def set_rate_limiter_verbose(verbose: bool) -> None:
    """Enable or disable verbose logging for all rate limiters."""
    from .clients.anthropic_client import (
        set_rate_limiter_verbose as set_anthropic_verbose,
    )
    from .clients.google_client import set_rate_limiter_verbose as set_gemini_verbose
    from .clients.mistral_client import set_rate_limiter_verbose as set_mistral_verbose
    from .clients.openai_client import set_rate_limiter_verbose as set_openai_verbose

    set_anthropic_verbose(verbose)
    set_gemini_verbose(verbose)
    set_openai_verbose(verbose)
    set_mistral_verbose(verbose)


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
        has_override = any(
            [
                openai_api_key,
                anthropic_api_key,
                mistral_api_key,
                gemini_api_key,
                openrouter_api_key,
                grok_api_key,
            ]
        )
        self._clients = self._build_clients() if has_override else None

    def _require_key(
        self,
        override: str | None,
        provider: str,
        env_vars: list[str],
        getter: Callable[[], str | None],
    ) -> str:
        return require_api_key(override or getter(), provider, env_vars)

    def _create_openai_client(self) -> OpenAI:
        api_key = self._require_key(
            self._openai_api_key, "openai", ["OPENAI_API_KEY"], get_openai_api_key
        )
        return OpenAI(api_key=api_key)

    def _create_openrouter_client(self) -> OpenAI:
        api_key = self._require_key(
            self._openrouter_api_key,
            "openrouter",
            ["OPENROUTER_API_KEY"],
            get_openrouter_api_key,
        )
        return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def _create_gemini_client(self) -> genai.Client:
        api_key = self._require_key(
            self._gemini_api_key,
            "gemini",
            ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            get_gemini_api_key,
        )
        return genai.Client(api_key=api_key)

    def _create_mistral_client(self) -> Mistral:
        api_key = self._require_key(
            self._mistral_api_key, "mistral", ["MISTRAL_API_KEY"], get_mistral_api_key
        )
        return Mistral(api_key=api_key)

    def _create_anthropic_client(self) -> Anthropic:
        api_key = self._require_key(
            self._anthropic_api_key,
            "anthropic",
            ["ANTHROPIC_API_KEY"],
            get_anthropic_api_key,
        )
        return Anthropic(api_key=api_key)

    def _create_grok_client(self) -> OpenAI:
        api_key = self._require_key(
            self._grok_api_key,
            "grok",
            ["XAI_API_KEY", "GROK_API_KEY"],
            get_grok_api_key,
        )
        return OpenAI(api_key=api_key, base_url=GROK_BASE_URL)

    def _build_clients(self) -> dict[str, Any]:
        return {
            "openai": LazyClient(self._create_openai_client, label="openai"),
            "openrouter": LazyClient(
                self._create_openrouter_client, label="openrouter"
            ),
            "gemini": LazyClient(self._create_gemini_client, label="gemini"),
            "mistral": LazyClient(self._create_mistral_client, label="mistral"),
            "anthropic": LazyClient(self._create_anthropic_client, label="anthropic"),
            "grok": LazyClient(self._create_grok_client, label="grok"),
        }

    def get_record_store(self) -> RecordStore:
        return self._record_store

    def _get_client(self, provider: str) -> Any | None:
        """Get client for provider, or None to use module-level default."""
        if self._clients is None:
            return None
        return self._clients.get(provider)

    def _get_provider(self, model: str) -> str:
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

    def ask_llm[T](
        self,
        user_msg: str,
        model: str,
        response_type: type[T] | None = None,
        sys_msg: str | None = None,
        *,
        max_parsing_retries: int = 2,
    ) -> T:
        """Route to appropriate provider and make LLM call.

        Args:
            user_msg: User message/prompt
            model: Model name - determines provider routing
            response_type: Type for structured output. Can be:
                - None or str: returns plain text
                - Pydantic model: returns model instance
                - list[X], tuple[...], etc.: wraps in Pydantic, then unwraps
            sys_msg: Optional system message
            max_parsing_retries: Retries for structured output parsing errors
        """
        provider = self._get_provider(model)
        client = self._get_client(provider)
        max_attempts = max_parsing_retries + 1

        # Import provider functions
        from .clients.anthropic_client import ask_anthropic
        from .clients.google_client import ask_gemini
        from .clients.grok_client import ask_grok
        from .clients.mistral_client import ask_mistral
        from .clients.openai_client import ask_openai
        from .clients.openrouter_client import ask_openrouter

        llm_fn = {
            "gemini": ask_gemini,
            "mistral": ask_mistral,
            "anthropic": ask_anthropic,
            "openrouter": ask_openrouter,
            "openai": ask_openai,
            "grok": ask_grok,
        }[provider]

        # Adapt response_type for LLM API (wrap if needed)
        adapter = ResponseTypeAdapter(response_type)
        llm_type = adapter.get_llm_type()

        for attempt in range(max_attempts):
            try:
                result = llm_fn(
                    user_msg=user_msg,
                    response_type=llm_type,
                    sys_msg=sys_msg,
                    model=model,
                    client_override=client,
                    record_store=self._record_store,
                )
                if llm_type not in (None, str):
                    try:
                        result = TypeAdapter(llm_type).validate_python(result)
                    except ValidationError as exc:
                        raise StructuredOutputParsingError(
                            "Structured LLM output did not match expected schema."
                        ) from exc

                return adapter.unwrap(result)
            except StructuredOutputParsingError:
                if attempt == max_attempts - 1:
                    raise

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
                    executor.submit(make_candidate_call, i)
                    for i in range(num_candidates)
                ]
                for future in as_completed(futures):
                    try:
                        candidates.append(future.result())
                    except Exception as e:
                        raise RuntimeError(f"Failed to generate candidate: {e}") from e
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


_default_client = Covenance(records_dir=get_env_records_dir())


def get_default_client() -> Covenance:
    return _default_client


def ask_llm[T](
    user_msg: str,
    model: str,
    response_type: type[T] | None = None,
    sys_msg: str | None = None,
    *,
    max_parsing_retries: int = 2,
) -> T:
    return _default_client.ask_llm(
        user_msg=user_msg,
        model=model,
        response_type=response_type,
        sys_msg=sys_msg,
        max_parsing_retries=max_parsing_retries,
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
