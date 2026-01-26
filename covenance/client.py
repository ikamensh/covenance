"""Instance-scoped LLM access with isolated keys and call records.

Module-level helpers route through the default instance so legacy API keeps working.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from google import genai
from mistralai import Mistral
from openai import OpenAI

from ._lazy_client import LazyClient
from .clients.openrouter_client import OPENROUTER_BASE_URL
from .keys import (
    get_anthropic_api_key,
    get_gemini_api_key,
    get_mistral_api_key,
    get_openai_api_key,
    get_openrouter_api_key,
    require_api_key,
)
from .client_context import use_instance_context
from .record import Record, RecordStore, get_default_record_store
from .unified import ask_llm as _ask_llm
from .unified import llm_consensus as _llm_consensus
from .usage import TokenUsage

class Covenance:
    """Isolated LLM configuration for per-label keys and call records."""

    def __init__(
        self,
        *,
        label: str | None = None,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        mistral_api_key: str | None = None,
        gemini_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        records_dir: str | Path | None = None,
        _record_store: RecordStore | None = None,
        _client_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.label = label
        self._openai_api_key = openai_api_key
        self._anthropic_api_key = anthropic_api_key
        self._mistral_api_key = mistral_api_key
        self._gemini_api_key = gemini_api_key
        self._openrouter_api_key = openrouter_api_key

        if _record_store is None:
            self._record_store = RecordStore(records_dir=records_dir, label=label)
        else:
            if records_dir is not None:
                raise ValueError("records_dir cannot be set when record_store is provided.")
            if label is not None and _record_store.label is None:
                _record_store.label = label
            self._record_store = _record_store

        if _client_overrides is not None:
            self._client_overrides = _client_overrides
        else:
            has_override = any(
                [
                    openai_api_key,
                    anthropic_api_key,
                    mistral_api_key,
                    gemini_api_key,
                    openrouter_api_key,
                ]
            )
            self._client_overrides = self._build_client_overrides() if has_override else None

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

    def _build_client_overrides(self) -> dict[str, Any]:
        return {
            "openai": LazyClient(self._create_openai_client, label="openai"),
            "openrouter": LazyClient(self._create_openrouter_client, label="openrouter"),
            "gemini": LazyClient(self._create_gemini_client, label="gemini"),
            "mistral": LazyClient(self._create_mistral_client, label="mistral"),
            "anthropic": LazyClient(self._create_anthropic_client, label="anthropic"),
        }

    def ask_llm[T](
        self,
        user_msg: str,
        model: str,
        format: type[T] | None = None,
        sys_msg: str | None = None,
        *,
        max_parsing_retries: int = 2,
    ) -> T:
        with use_instance_context(self._record_store, self._client_overrides):
            return _ask_llm(
                user_msg=user_msg,
                model=model,
                format=format,
                sys_msg=sys_msg,
                max_parsing_retries=max_parsing_retries,
            )

    def llm_consensus[T](
        self,
        user_msg: str,
        model: str,
        format: type[T] | None = None,
        sys_msg: str | None = None,
        *,
        num_candidates: int = 3,
        additional_models: list[str] | None = None,
        integration_model: str | None = None,
        parallel: bool = True,
    ) -> T:
        with use_instance_context(self._record_store, self._client_overrides):
            return _llm_consensus(
                user_msg=user_msg,
                model=model,
                format=format,
                sys_msg=sys_msg,
                num_candidates=num_candidates,
                additional_models=additional_models,
                integration_model=integration_model,
                parallel=parallel,
            )

    def record_llm_call(
        self,
        *,
        model: str,
        provider: str,
        usage: TokenUsage,
        started_at: datetime,
        ended_at: datetime,
        tpm_retry_wait_seconds: float = 0.0,
    ) -> None:
        from .metrics import record_llm_call

        with use_instance_context(self._record_store, self._client_overrides):
            record_llm_call(
                model=model,
                provider=provider,
                usage=usage,
                started_at=started_at,
                ended_at=ended_at,
                tpm_retry_wait_seconds=tpm_retry_wait_seconds,
            )

    def get_records(self) -> list[Record]:
        return self._record_store.get_records()

    def clear_records(self) -> None:
        self._record_store.clear_records()

    def set_llm_call_records_dir(self, path: str | Path | None) -> None:
        self._record_store.set_llm_call_records_dir(path)

    def get_llm_call_records_dir(self) -> Path | None:
        return self._record_store.get_llm_call_records_dir()

    def get_llm_call_records_path(self) -> Path | None:
        return self._record_store.get_llm_call_records_path()


_default_client = Covenance(
    _record_store=get_default_record_store(),
    _client_overrides=None,
)


def get_default_client() -> Covenance:
    return _default_client


def ask_llm[T](
    user_msg: str,
    model: str,
    format: type[T] | None = None,
    sys_msg: str | None = None,
    *,
    max_parsing_retries: int = 2,
) -> T:
    return _default_client.ask_llm(
        user_msg=user_msg,
        model=model,
        format=format,
        sys_msg=sys_msg,
        max_parsing_retries=max_parsing_retries,
    )


def llm_consensus[T](
    user_msg: str,
    model: str,
    format: type[T] | None = None,
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
        format=format,
        sys_msg=sys_msg,
        num_candidates=num_candidates,
        additional_models=additional_models,
        integration_model=integration_model,
        parallel=parallel,
    )

