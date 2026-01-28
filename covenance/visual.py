from __future__ import annotations

import re
from datetime import datetime

from covenance import Record, get_records


def _shorten_model_name(model: str, max_len: int = 13) -> str:
    """Shorten model name while preserving provider identity and distinguishing features.

    Strategy: abbreviate progressively until it fits.
    1. Shorten provider prefix (gemini- → g, claude- → c, gpt- stays)
    2. Remove date suffixes
    3. If still too long, abbreviate size suffixes (-lite → -l, -mini → -m)
    4. If still too long, abbreviate variants (-flash → -f, -sonnet → -s)

    Examples:
        gemini-2.5-flash-lite → g2.5-flash-l (12 chars)
        gemini-2.5-flash → g2.5-flash (10 chars)
        gpt-4.1-nano → gpt-4.1-nano (12 chars)
        claude-sonnet-4-20250514 → c-sonnet-4 (10 chars)
    """
    # Step 1: Shorten long provider prefixes (keep gpt-/grok- as is, they're short)
    provider_abbrev = [
        ("gemini-", "g"),
        ("claude-", "c"),
        ("mistral-", "mi-"),
        ("codestral-", "co-"),
    ]
    for prefix, abbrev in provider_abbrev:
        if model.startswith(prefix):
            model = abbrev + model[len(prefix) :]
            break

    # Step 2: Remove date suffixes like -20250514
    model = re.sub(r"-\d{8}$", "", model)

    # Step 3: If too long, abbreviate size suffixes
    if len(model) > max_len:
        model = model.replace("-lite", "-l").replace("-mini", "-m")

    # Step 4: If still too long, abbreviate variant names
    if len(model) > max_len:
        model = (
            model.replace("-flash", "-f")
            .replace("-sonnet", "-s")
            .replace("-opus", "-o")
            .replace("-haiku", "-h")
        )

    # Final truncation if still too long
    if len(model) > max_len:
        model = model[: max_len - 1] + "…"

    return model


def print_call_timeline(
    records: list[Record] | None = None,
    width: int = 80,
) -> None:
    """Print a terminal-based waterfall chart of LLM call timings.

    Each line represents one call, sorted by start time (top to bottom).
    Block characters show when each call was active on a shared time axis,
    making parallel calls and sequence visible at a glance.
    """
    if records is None:
        records = get_records()

    if not records:
        print("\nNo LLM calls to display.")
        return

    # Parse timestamps and build call data
    calls: list[tuple[datetime, datetime, Record]] = []
    for r in records:
        start = datetime.fromisoformat(r.started_at)
        end = datetime.fromisoformat(r.ended_at)
        calls.append((start, end, r))

    # Find time range
    min_time = min(c[0] for c in calls)
    max_time = max(c[1] for c in calls)
    total_seconds = (max_time - min_time).total_seconds()

    if total_seconds == 0:
        total_seconds = 0.1  # Avoid division by zero for instant calls

    # Sort by start time
    calls.sort(key=lambda c: c[0])

    # Header
    total_dur_str = _format_duration(total_seconds)
    print(f"\nLLM Call Timeline ({total_dur_str} total, {len(records)} calls)")

    # Time axis labels
    left_label = "0s"
    right_label = total_dur_str
    axis_content_width = width - len(left_label) - len(right_label)
    axis_line = (
        f"  {'':13} {'':>6}  |{left_label}{' ' * axis_content_width}{right_label}|"
    )
    print(axis_line)

    # Each call as a line
    for start, end, record in calls:
        # Calculate bar position
        start_offset = (start - min_time).total_seconds() / total_seconds
        end_offset = (end - min_time).total_seconds() / total_seconds

        start_col = int(start_offset * width)
        end_col = max(start_col + 1, int(end_offset * width))

        # Build bar: spaces before, blocks during, spaces after
        bar = " " * start_col + "█" * (end_col - start_col) + " " * (width - end_col)

        # Format model name
        model = _shorten_model_name(record.model).ljust(13)

        # Format duration
        dur_str = _format_duration(record.duration_seconds).rjust(6)

        print(f"  {model} {dur_str}  |{bar}|")


def _format_duration(seconds: float) -> str:
    """Format duration for display: 1.2s, 45s, 1m30s, 5m, etc."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if secs == 0:
        return f"{minutes}m"
    return f"{minutes}m{secs}s"
