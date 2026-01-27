"""Unified LLM client for OpenAI, Google Gemini, Mistral, Anthropic Claude, and OpenRouter."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

from .record import Record
from .client import (
    Covenance,
    ask_llm,
    get_default_client,
    llm_consensus,
    set_rate_limiter_verbose,
)
from .record import (
    clear_records,
    get_llm_call_records_dir,
    get_records,
    set_llm_call_records_dir,
)
from .usage import TokenUsage, usage_stats

__all__ = [
    "__version__",
    "ask_llm",
    "llm_consensus",
    "Covenance",
    "get_default_client",
    "set_rate_limiter_verbose",
    "TokenUsage",
    "usage_stats",
    # Call records
    "Record",
    "get_records",
    "clear_records",
    "get_llm_call_records_dir",
    "set_llm_call_records_dir",
]
