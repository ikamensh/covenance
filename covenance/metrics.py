"""LLM operation context with metrics collection.

This module provides scoped context for LLM operations, including:
- Business context (session_id, task name, etc.) for logging
- Metrics collection (token usage, timing) for DB persistence
"""

from __future__ import annotations

import contextvars
import logging
import os
from datetime import datetime
from threading import Lock
from typing import Any

from .llm_calls import Record
from .llm_calls import record_llm_call as _log_llm_call
from .usage import TokenUsage

logger = logging.getLogger(__name__)


class LLMOperationContext:
    """Thread-safe context for LLM operations with metrics collection.

    Provides:
    - Business context (who's calling and why) for structured logging
    - Metrics collection for DB persistence

    Usage:
        # In BackendOperation workers:
        with LLMOperationContext.start(session_id="abc-123", operation="dpia_creation"):
            result = generate_dpia(...)

        # In standalone scripts:
        with LLMOperationContext.start(task="prompt_translation", lang="DE"):
            translate_prompts(...)

        # All LLM calls within the context are logged with business context
        # and metrics are collected for later retrieval via ctx.to_dict()
    """

    _current: contextvars.ContextVar[LLMOperationContext | None] = (
        contextvars.ContextVar("llm_operation_context", default=None)
    )

    def __init__(self, context: dict[str, Any] | None = None):
        self._lock = Lock()
        self._records: list[Record] = []
        self.context: dict[str, Any] = context or {}

    @classmethod
    def start(cls, **context: Any) -> LLMOperationContext:
        """Create and set as current context.

        Args:
            **context: Business context key-value pairs, e.g.:
                - session_id: DPIA session ID
                - operation_id: Backend operation ID
                - task: Script/task name
                - Any other relevant context

        Returns:
            The new context (use as context manager or call stop() when done)
        """
        ctx = cls(context=context if context else None)
        cls._current.set(ctx)
        return ctx

    @classmethod
    def current(cls) -> LLMOperationContext | None:
        """Get the current context, if any."""
        return cls._current.get()

    @classmethod
    def set_current(cls, ctx: LLMOperationContext | None) -> None:
        """Set current context (for thread propagation to ThreadPoolExecutor workers)."""
        cls._current.set(ctx)

    def stop(self) -> dict[str, Any]:
        """Stop this context and return collected metrics as dict."""
        self._current.set(None)
        return self.to_dict()

    def get_context_str(self) -> str:
        """Format context as string for logging, e.g. 'session_id=abc task=translate'."""
        if not self.context:
            return ""
        return " ".join(f"{k}={v}" for k, v in self.context.items())

    def record_call(
        self,
        *,
        model: str,
        provider: str,
        usage: TokenUsage,
        started_at: datetime,
        ended_at: datetime,
        tpm_retry_wait_seconds: float = 0.0,
    ) -> None:
        """Record a single LLM call."""
        record = _log_llm_call(
            model=model,
            provider=provider,
            usage=usage,
            started_at=started_at,
            ended_at=ended_at,
            tpm_retry_wait_seconds=tpm_retry_wait_seconds,
        )
        self._append_record(record)

    def _append_record(self, record: Record) -> None:
        with self._lock:
            self._records.append(record)

    def get_records(self) -> list[Record]:
        """Get copy of all recorded calls."""
        with self._lock:
            return self._records.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for DB storage."""
        host = os.environ.get("K_SERVICE", "local")
        with self._lock:
            return {
                "host": host,
                "calls": [r.model_dump() for r in self._records],
            }

    def __enter__(self) -> LLMOperationContext:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._current.set(None)


# Backwards compatibility alias
MetricsContext = LLMOperationContext


def record_llm_call(
    *,
    model: str,
    provider: str,
    usage: TokenUsage,
    started_at: datetime,
    ended_at: datetime,
    tpm_retry_wait_seconds: float = 0.0,
) -> None:
    """Record an LLM call to the current context (if active) and log it.

    If no context is active, still logs the call but without business context.
    """
    duration = (ended_at - started_at).total_seconds()

    record = _log_llm_call(
        model=model,
        provider=provider,
        usage=usage,
        started_at=started_at,
        ended_at=ended_at,
        tpm_retry_wait_seconds=tpm_retry_wait_seconds,
    )
    ctx = LLMOperationContext.current()
    if ctx is not None:
        ctx._append_record(record)
        # Log with business context
        ctx_str = ctx.get_context_str()
        if ctx_str:
            logger.info(
                f"{ctx_str}: LLM call {provider}/{model} "
                f"tokens={usage.total_tokens} (in={usage.prompt_tokens}, out={usage.completion_tokens}, cached={usage.cached_tokens}) "
                f"duration={duration:.2f}s"
            )
        else:
            logger.info(
                f"LLM call {provider}/{model} "
                f"tokens={usage.total_tokens} (in={usage.prompt_tokens}, out={usage.completion_tokens}, cached={usage.cached_tokens}) "
                f"duration={duration:.2f}s"
            )
    else:
        # No context, still log but note it's untracked
        logger.info(
            f"[no-context] LLM call {provider}/{model} "
            f"tokens={usage.total_tokens} duration={duration:.2f}s"
        )
