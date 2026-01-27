"""Export persisted LLM call records to a JSON file for visualization."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from covenance.record import (
    Record,
    get_llm_call_records_path,
    set_llm_call_records_dir,
)

# ---------------------------
# Input configuration
# ---------------------------

# Optional override of the persisted records directory.
# If None, uses the COVENANCE_RECORDS_DIR environment variable.
CALL_RECORDS_DIR: Path | None = None

# Output JSON file for the UI.
OUTPUT_FILE = Path(__file__).parent / "generated/llm_calls.json"


class LLMCallRecordsExport(BaseModel):
    """Export payload for UI visualization."""

    query_time: datetime
    count: int
    records: list[Record]


def _resolve_records_path() -> Path:
    if CALL_RECORDS_DIR is not None:
        set_llm_call_records_dir(CALL_RECORDS_DIR)
    records_path = get_llm_call_records_path()
    if records_path is None:
        raise RuntimeError(
            "Please set CALL_RECORDS_DIR or COVENANCE_RECORDS_DIR environment variable "
            "to locate records (you can also use a .env file)."
        )
    return records_path


def _load_records(records_path: Path) -> list[Record]:
    if not records_path.exists():
        raise FileNotFoundError(f"No LLM call records found at {records_path}")
    records: list[Record] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(Record.model_validate_json(stripped))
    records.sort(key=lambda record: record.started_at)
    return records


def export_llm_call_records() -> None:
    records_path = _resolve_records_path()
    records = _load_records(records_path)

    payload = LLMCallRecordsExport(
        query_time=datetime.now(UTC),
        count=len(records),
        records=records,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved {len(records)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    export_llm_call_records()
