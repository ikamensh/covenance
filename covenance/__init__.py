"""Online LLM clients for OpenAI, Google Gemini, Mistral, Anthropic Claude, and OpenRouter."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

from covenance.clients.anthropic_client import ClaudeModels, ask_anthropic
from covenance.clients.google_client import GeminiModels, ask_gemini
from covenance.clients.mistral_client import MistralModels, ask_mistral
from covenance.clients.openai_client import OpenaiModels, ask_openai
from covenance.clients.openrouter_client import (
    OpenRouterModels,
    ask_openrouter,
)

from .metrics import (
    LLMOperationContext,
    MetricsContext,  # Backwards compat alias
    Record,
    record_llm_call,
)
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
    "ask_anthropic",
    "ask_gemini",
    "ask_openai",
    "ask_mistral",
    "ask_openrouter",
    "ask_llm",
    "llm_consensus",
    "Covenance",
    "get_default_client",
    "set_rate_limiter_verbose",
    "ClaudeModels",
    "GeminiModels",
    "MistralModels",
    "OpenaiModels",
    "OpenRouterModels",
    "TokenUsage",
    "usage_stats",
    # LLM operation context and metrics collection
    "Record",
    "LLMOperationContext",
    "MetricsContext",  # Backwards compat alias
    "record_llm_call",
    "get_records",
    "clear_records",
    "get_llm_call_records_dir",
    "set_llm_call_records_dir",
]
