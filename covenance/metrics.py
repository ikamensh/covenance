"""LLM call recording with logging."""

from __future__ import annotations

import logging
from datetime import datetime

from .record import RecordStore
from .usage import TokenUsage

logger = logging.getLogger(__name__)


def _get_default_record_store() -> RecordStore:
    """Get the default record store from the default client."""
    from .client import get_default_client

    return get_default_client().get_record_store()


def record_llm_call(
    *,
    model: str,
    provider: str,
    usage: TokenUsage,
    started_at: datetime,
    ended_at: datetime,
    tpm_retry_wait_seconds: float = 0.0,
    record_store: RecordStore | None = None,
) -> None:
    """Record an LLM call to the given store (or default) and log it."""
    duration = (ended_at - started_at).total_seconds()
    store = record_store or _get_default_record_store()

    store.record_llm_call(
        model=model,
        provider=provider,
        usage=usage,
        started_at=started_at,
        ended_at=ended_at,
        tpm_retry_wait_seconds=tpm_retry_wait_seconds,
    )
    logger.info(
        f"LLM call {provider}/{model} "
        f"tokens={usage.total_tokens} (in={usage.prompt_tokens}, out={usage.completion_tokens}, cached={usage.cached_tokens}) "
        f"duration={duration:.2f}s"
    )
