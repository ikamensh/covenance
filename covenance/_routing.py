"""Provider routing logic shared across backends."""

from typing import Literal

Backend = Literal["native", "pydantic"]

VALID_BACKENDS = {"native", "pydantic"}

_PROVIDER_FIELDS = ("openai", "grok", "gemini", "anthropic", "mistral", "openrouter")

# Defaults: native SDK is used where it outperforms pydantic-ai (determined by stress testing)
_DEFAULTS: dict[str, str] = {
    "openai": "native",
    "grok": "native",
    "gemini": "pydantic",
    "anthropic": "pydantic",
    "mistral": "pydantic",
    "openrouter": "pydantic",
}


class Backends:
    """Per-provider backend selection.

    Each attribute is the backend for that provider: "native" or "pydantic".
    Defaults are determined by stress testing — native SDK is used where it
    outperforms pydantic-ai for structured output reliability.
    """

    openai: Backend
    grok: Backend
    gemini: Backend
    anthropic: Backend
    mistral: Backend
    openrouter: Backend

    def __init__(self) -> None:
        for provider, backend in _DEFAULTS.items():
            object.__setattr__(self, provider, backend)

    def __setattr__(self, name: str, value: str) -> None:
        if name not in _PROVIDER_FIELDS:
            raise AttributeError(
                f"No provider {name!r}. Valid providers: {', '.join(_PROVIDER_FIELDS)}"
            )
        if value not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend {value!r} for {name}, must be 'native' or 'pydantic'"
            )
        object.__setattr__(self, name, value)

    def set_all(self, backend: Backend) -> None:
        """Set all providers to the same backend."""
        for provider in _PROVIDER_FIELDS:
            setattr(self, provider, backend)

    def get(self, provider: str) -> str:
        """Get backend for a provider, defaulting to 'pydantic' for unknown providers."""
        return getattr(self, provider, "pydantic")

    def __repr__(self) -> str:
        native = [p for p in _PROVIDER_FIELDS if getattr(self, p) == "native"]
        pydantic = [p for p in _PROVIDER_FIELDS if getattr(self, p) == "pydantic"]
        parts = []
        if native:
            parts.append(f"native=[{', '.join(native)}]")
        if pydantic:
            parts.append(f"pydantic=[{', '.join(pydantic)}]")
        return f"Backends({', '.join(parts)})"


# Backward compat
NATIVE_BACKEND_PROVIDERS = {p for p, b in _DEFAULTS.items() if b == "native"}
DEFAULT_BACKENDS = _DEFAULTS


def get_provider(model: str) -> str:
    """Determine provider from model name."""
    if model.startswith("gemini"):
        return "gemini"
    elif model.startswith(("mistral", "ministral", "codestral")):
        return "mistral"
    elif model.startswith("claude"):
        return "anthropic"
    elif model.startswith("grok"):
        return "grok"
    elif "/" in model:
        return "openrouter"
    else:
        return "openai"
