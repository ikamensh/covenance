"""Always-on LLM call logging with optional local persistence."""

from __future__ import annotations

import inspect
import logging
import os
from datetime import datetime
from pathlib import Path
from threading import Lock

from pydantic import BaseModel

from .usage import TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_RECORDS_FILENAME = "llm_call_records.jsonl"
RECORDS_DIR_ENV = "COVENANCE_RECORDS_DIR"


class Record(BaseModel):
    """Record of a single LLM API call."""

    model: str
    provider: str
    tokens_input: int
    tokens_output: int
    tokens_cached: int = 0
    tokens_total: int
    duration_seconds: float
    tpm_retry_wait_seconds: float = 0.0
    started_at: str  # ISO 8601 timestamp
    ended_at: str  # ISO 8601 timestamp
    # Caller info (best-effort, for debugging)
    caller_function: str | None = None
    caller_file: str | None = None
    caller_line: int | None = None


class RecordStore:
    """Thread-safe in-memory LLM call log with optional JSONL persistence."""

    def __init__(
        self,
        records_dir: str | Path | None = None,
        *,
        label: str | None = None,
        records_filename: str = DEFAULT_RECORDS_FILENAME,
    ) -> None:
        self._records: list[Record] = []
        self._lock = Lock()
        self._records_dir: Path | None = None
        self._records_filename = records_filename
        self.label = label
        if records_dir is not None:
            self.set_llm_call_records_dir(records_dir)

    def set_llm_call_records_dir(self, path: str | Path | None) -> None:
        """Enable or disable persistence of call records to a local folder."""
        if path is None:
            self._records_dir = None
            return
        self._records_dir = Path(path).expanduser().resolve()

    def get_llm_call_records_dir(self) -> Path | None:
        """Return the configured directory for local call record persistence."""
        return self._records_dir

    def get_llm_call_records_path(self) -> Path | None:
        """Return the JSONL file path for persisted call records, if enabled."""
        if self._records_dir is None:
            return None
        return self._records_dir / self._records_filename

    def record_llm_call(
        self,
        *,
        model: str,
        provider: str,
        usage: TokenUsage,
        started_at: datetime,
        ended_at: datetime,
        tpm_retry_wait_seconds: float = 0.0,
        caller_function: str | None = None,
        caller_file: str | None = None,
        caller_line: int | None = None,
    ) -> Record:
        """Record a single LLM call in memory and optionally persist it to disk."""
        duration_seconds = round((ended_at - started_at).total_seconds(), 3)
        record = Record(
            model=model,
            provider=provider,
            tokens_input=usage.prompt_tokens,
            tokens_output=usage.completion_tokens,
            tokens_cached=usage.cached_tokens,
            tokens_total=usage.total_tokens,
            duration_seconds=duration_seconds,
            tpm_retry_wait_seconds=tpm_retry_wait_seconds,
            started_at=started_at.isoformat(),
            ended_at=ended_at.isoformat(),
            caller_function=caller_function,
            caller_file=caller_file,
            caller_line=caller_line,
        )
        with self._lock:
            self._records.append(record)
            self._persist_record(record)
        return record

    def get_records(self) -> list[Record]:
        """Return a copy of all call records captured in this process."""
        with self._lock:
            return self._records.copy()

    def clear_records(self) -> None:
        """Clear in-memory call records (does not delete persisted files)."""
        with self._lock:
            self._records.clear()

    def _persist_record(self, record: Record) -> None:
        if self._records_dir is None:
            return
        self._records_dir.mkdir(parents=True, exist_ok=True)
        output_file = self._records_dir / self._records_filename
        with output_file.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")


def get_env_records_dir() -> str | None:
    """Return the persisted records directory from env, if configured."""
    from .keys import load_env_if_present

    load_env_if_present()
    return os.getenv(RECORDS_DIR_ENV)


def _default_client():
    from .client import get_default_client

    return get_default_client()


def set_llm_call_records_dir(path: str | Path | None) -> None:
    """Enable or disable persistence of call records to a local folder."""
    _default_client().set_llm_call_records_dir(path)


def get_llm_call_records_dir() -> Path | None:
    """Return the configured directory for local call record persistence."""
    return _default_client().get_llm_call_records_dir()


def get_llm_call_records_path() -> Path | None:
    """Return the JSONL file path for persisted call records, if enabled."""
    return _default_client().get_llm_call_records_path()


def get_records() -> list[Record]:
    """Return a copy of all call records captured in this process."""
    return _default_client().get_records()


def clear_records() -> None:
    """Clear in-memory call records (does not delete persisted files)."""
    _default_client().clear_records()


def _get_caller_info(skip_frames: int = 4) -> tuple[str | None, str | None, int | None]:
    """Extract caller info from the call stack.
    
    Returns (function_name, filename, lineno) of the caller.
    Best-effort: returns (None, None, None) if stack is too short.
    """
    stack = inspect.stack()
    if len(stack) > skip_frames:
        frame = stack[skip_frames]
        filepath = Path(frame.filename)
        return frame.function, filepath.name, frame.lineno
    return None, None, None


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
    store = record_store or _default_client().get_record_store()
    
    caller_function, caller_file, caller_line = _get_caller_info()

    store.record_llm_call(
        model=model,
        provider=provider,
        usage=usage,
        started_at=started_at,
        ended_at=ended_at,
        tpm_retry_wait_seconds=tpm_retry_wait_seconds,
        caller_function=caller_function,
        caller_file=caller_file,
        caller_line=caller_line,
    )
    logger.info(
        f"LLM call {provider}/{model} "
        f"tokens={usage.total_tokens} (in={usage.prompt_tokens}, out={usage.completion_tokens}, cached={usage.cached_tokens}) "
        f"duration={duration:.2f}s"
    )

