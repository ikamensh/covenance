"""Online LLM clients for OpenAI, Google Gemini, Mistral, Anthropic Claude, and OpenRouter."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

from covenance.clients.anthropic_client import ClaudeModels, ask_anthropic_structured
from covenance.clients.google_client import GeminiModels, ask_gemini_structured
from covenance.clients.mistral_client import MistralModels, ask_mistral_structured
from covenance.clients.openai_client import OpenaiModels, ask_chatgpt_structured
from covenance.clients.openrouter_client import (
    OpenRouterModels,
    ask_openrouter_structured,
)

from .metrics import (
    LLMOperationContext,
    MetricsContext,  # Backwards compat alias
    Record,
    record_llm_call,
)
from .record import (
    get_llm_call_records_dir,
    get_records,
    set_llm_call_records_dir,
)
from .unified import (
    ask_llm,
    llm_consensus,
)
from .usage import TokenUsage, usage_stats

__all__ = [
    "__version__",
    "ask_anthropic_structured",
    "ask_gemini_structured",
    "ask_chatgpt_structured",
    "ask_mistral_structured",
    "ask_openrouter_structured",
    "ask_llm",  # Unified wrapper
    "llm_consensus",  # Multi-call with integration
    "ClaudeModels",
    "GeminiModels",
    "MistralModels",
    "OpenaiModels",
    "OpenRouterModels",
    "TokenUsage",
    "usage_stats",  # Global usage statistics tracker
    # LLM operation context and metrics collection
    "Record",
    "LLMOperationContext",
    "MetricsContext",  # Backwards compat alias
    "record_llm_call",
    "get_records",
    "get_llm_call_records_dir",
    "set_llm_call_records_dir",
]
