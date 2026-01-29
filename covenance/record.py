"""Always-on LLM call logging with optional local persistence."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from threading import Lock

from pydantic import BaseModel

from ._caller_context import get_caller_info

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
    cost_usd: float | None = None  # None if pricing unknown for this model
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
        from covenance.pricing import calculate_cost

        duration_seconds = round((ended_at - started_at).total_seconds(), 3)
        cost_usd = calculate_cost(
            model=model,
            provider=provider,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
        )
        record = Record(
            model=model,
            provider=provider,
            tokens_input=usage.prompt_tokens,
            tokens_output=usage.completion_tokens,
            tokens_cached=usage.cached_tokens,
            tokens_total=usage.total_tokens,
            cost_usd=cost_usd,
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


def set_records_dir(path: str | Path | None) -> None:
    """Enable or disable persistence of call records to a local folder."""
    from .client import _default_client

    _default_client.get_record_store()._records_dir = path


def get_records_dir() -> Path | None:
    """Return the configured directory for local call record persistence."""
    from .client import _default_client

    return _default_client.get_record_store()._records_dir


def get_llm_call_records_path() -> Path | None:
    """Return the JSONL file path for persisted call records, if enabled."""
    from .client import _default_client

    return _default_client.get_record_store().get_llm_call_records_path()


def get_records() -> list[Record]:
    """Return a copy of all call records captured in this process."""
    from .client import _default_client

    return _default_client.get_records()


def clear_records() -> None:
    """Clear in-memory call records (does not delete persisted files)."""
    from .client import _default_client

    _default_client.clear_records()


class TokenUsage(BaseModel):
    """Standardized token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0  # Tokens read from cache (provider-specific support)


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
    from .client import _default_client

    duration = (ended_at - started_at).total_seconds()
    store = record_store or _default_client.get_record_store()

    caller_function, caller_file, caller_line = get_caller_info()

    record = store.record_llm_call(
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
    cost_str = f"${record.cost_usd:.6f}" if record.cost_usd is not None else "n/a"
    logger.info(
        f"LLM call {provider}/{model} "
        f"tokens={usage.total_tokens} (in={usage.prompt_tokens}, out={usage.completion_tokens}, cached={usage.cached_tokens}) "
        f"cost={cost_str} duration={duration:.2f}s"
    )


def usage_summary(records: list[Record] | None = None) -> dict:
    """Compute usage summary from records.

    Args:
        records: List of Record objects. If None, uses get_records().

    Returns:
        Dict with keys: calls, tokens_input, tokens_output, tokens_cached, tokens_total,
        cost_usd, models (set of "provider/model" strings), has_openrouter (bool).
    """
    if records is None:
        records = get_records()

    total_cost = 0.0
    total_input = 0
    total_output = 0
    total_cached = 0
    models_used: set[str] = set()
    has_openrouter = False

    for record in records:
        if record.cost_usd is not None:
            total_cost += record.cost_usd
        total_input += record.tokens_input
        total_output += record.tokens_output
        total_cached += record.tokens_cached
        models_used.add(f"{record.provider}/{record.model}")
        if record.provider == "openrouter":
            has_openrouter = True

    return {
        "calls": len(records),
        "tokens_input": total_input,
        "tokens_output": total_output,
        "tokens_cached": total_cached,
        "tokens_total": total_input + total_output,
        "cost_usd": total_cost,
        "models": models_used,
        "has_openrouter": has_openrouter,
    }


def print_usage(
    records: list[Record] | None = None,
    title: str = "LLM Usage Summary",
    cost_format: str = "plain",
) -> None:
    """Print a formatted usage summary to stdout.

    Args:
        records: List of Record objects. If None, uses get_records().
        title: Header title for the summary block.
        cost_format: How to format costs. Options:
            - "plain": Always show dollars with 2 decimals (default)
            - "cent": Show cents with 3 decimals for costs < $0.01, dollars otherwise
            - "exponential": Show exponential notation for costs < $0.01, dollars otherwise
    """
    summary = usage_summary(records)

    if summary["calls"] == 0:
        print(f"\n{title}: No LLM calls recorded.")
        return

    print(f"\n{'=' * 50}")
    print(title)
    print("=" * 50)
    print(f"  Calls: {summary['calls']}")
    tokens_input = summary["tokens_input"]
    tokens_cached = summary["tokens_cached"]
    tokens_output = summary["tokens_output"]

    if tokens_cached > 0:
        tokens_new = tokens_input - tokens_cached
        in_part = f"In: {tokens_new:,} new + {tokens_cached:,} cached"
    else:
        in_part = f"In: {tokens_input:,}"

    print(f"  Tokens: {summary['tokens_total']:,} ({in_part}, Out: {tokens_output:,})")

    cost_usd = summary["cost_usd"]
    cost_line = "  Cost: "
    if cost_format == "cent" and cost_usd < 0.01:
        cost_cents = cost_usd * 100
        cost_line += f"{cost_cents:.3f}¢"
    elif cost_format == "exponential" and cost_usd < 0.01:
        cost_line += f"${cost_usd:.2e}"
    else:
        # Plain format: show 4 decimals for small numbers, 2 decimals otherwise
        if cost_usd > 0 and cost_usd < 0.01:
            cost_line += f"${cost_usd:.4f}"
        else:
            cost_line += f"${cost_usd:.2f}"
    
    if summary.get("has_openrouter", False):
        cost_line += " (excluding OpenRouter calls)"
    
    print(cost_line)

    print(f"  Models: {', '.join(sorted(summary['models']))}")


def load_records_from_jsonl(path: str | Path) -> list[Record]:
    """Load LLM call records from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No records file found at {path}")
    records: list[Record] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(Record.model_validate_json(stripped))
    records.sort(key=lambda r: r.started_at)
    return records
