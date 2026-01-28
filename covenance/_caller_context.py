"""Caller context tracking for LLM call records.

Captures the external caller location before spawning threads, allowing
record_llm_call() to attribute calls to the correct source location even
when executed in thread pools.
"""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from pathlib import Path

# Context variable for caller info - allows propagating caller location across threads
_caller_info_ctx: ContextVar[tuple[str | None, str | None, int | None] | None] = (
    ContextVar("caller_info", default=None)
)


def _get_caller_info_from_stack() -> tuple[str | None, str | None, int | None]:
    """Walk stack to find first frame outside covenance package."""
    stack = inspect.stack()
    covenance_dir = Path(__file__).parent.resolve()

    for frame in stack[1:]:  # skip this function itself
        frame_path = Path(frame.filename).resolve()
        try:
            frame_path.relative_to(covenance_dir)
        except ValueError:
            # Not inside covenance - this is the external caller
            return frame.function, frame_path.name, frame.lineno

    return None, None, None


def capture_caller_context() -> None:
    """Capture current external caller and store in context variable.

    Call this at entry points (like llm_consensus) before spawning threads.
    Use copy_context() to propagate to threads via executor.submit(ctx.run, fn, args).
    """
    _caller_info_ctx.set(_get_caller_info_from_stack())


def get_caller_info() -> tuple[str | None, str | None, int | None]:
    """Get caller info from context var if set, otherwise walk stack."""
    ctx_info = _caller_info_ctx.get()
    if ctx_info is not None:
        return ctx_info
    return _get_caller_info_from_stack()
