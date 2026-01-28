"""Unified LLM client for OpenAI, Google Gemini, Mistral, Anthropic Claude, and OpenRouter."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

from .client import (
    Covenance,
    ask_llm,
    llm_consensus,
)
from .record import (
    Record,
    clear_records,
    get_records,
    get_records_dir,
    load_records_from_jsonl,
    print_usage,
    set_records_dir,
    usage_summary,
)
from .visual import print_call_timeline

__all__ = [
    "__version__",
    "ask_llm",
    "llm_consensus",
    "Covenance",
    # Call records
    "Record",
    "get_records",
    "clear_records",
    "get_records_dir",
    "set_records_dir",
    "load_records_from_jsonl",
    "usage_summary",
    "print_usage",
    "print_call_timeline",
]
