"""Always-on LLM call logging with optional local persistence."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from threading import Lock

from pydantic import BaseModel

from .usage import TokenUsage
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


_records: list[Record] = []
_lock = Lock()
_records_dir: Path | None = None


def set_llm_call_records_dir(path: str | Path | None) -> None:
    """Enable or disable persistence of call records to a local folder."""
    global _records_dir
    if path is None:
        _records_dir = None
        return
    _records_dir = Path(path).expanduser().resolve()


def get_llm_call_records_dir() -> Path | None:
    """Return the configured directory for local call record persistence."""
    return _records_dir


def get_llm_call_records_path() -> Path | None:
    """Return the JSONL file path for persisted call records, if enabled."""
    if _records_dir is None:
        return None
    return _records_dir / DEFAULT_RECORDS_FILENAME


def record_llm_call(
    *,
    model: str,
    provider: str,
    usage: TokenUsage,
    started_at: datetime,
    ended_at: datetime,
    tpm_retry_wait_seconds: float = 0.0,
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
    )
    with _lock:
        _records.append(record)
        _persist_record(record)
    return record


def get_records() -> list[Record]:
    """Return a copy of all call records captured in this process."""
    with _lock:
        return _records.copy()


def clear_records() -> None:
    """Clear in-memory call records (does not delete persisted files)."""
    with _lock:
        _records.clear()


def _persist_record(record: Record) -> None:
    if _records_dir is None:
        return
    _records_dir.mkdir(parents=True, exist_ok=True)
    output_file = _records_dir / DEFAULT_RECORDS_FILENAME
    with output_file.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json())
        handle.write("\n")


from .keys import load_env_if_present

load_env_if_present()
_env_records_dir = os.getenv(RECORDS_DIR_ENV)
if _env_records_dir:
    set_llm_call_records_dir(_env_records_dir)

