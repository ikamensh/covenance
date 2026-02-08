"""Common result type returned by all LLM backends."""

from dataclasses import dataclass
from typing import Any

from .record import TokenUsage


@dataclass
class BackendResult:
    """Result from an LLM backend call, used for unified recording.

    Both pydantic-ai and native backends return this type, allowing
    client.py to record calls uniformly regardless of backend.
    """

    output: Any  # The LLM response (str, parsed model, etc.)
    usage: TokenUsage
    tpm_retries: int = 0
    tpm_wait_seconds: float = 0.0
    structured_output_retries: int = 0
