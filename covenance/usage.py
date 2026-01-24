"""Shared types and utilities for LLM clients."""

import inspect
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Standardized token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0  # Tokens read from cache (provider-specific support)


@dataclass(frozen=True)
class CallerInfo:
    """Information about a function caller."""

    filename: str
    function: str
    lineno: int
    module: str | None = None

    def __str__(self) -> str:
        """Return a human-readable representation."""
        if self.module:
            return f"{self.module}.{self.function} ({self.filename}:{self.lineno})"
        return f"{self.function} ({self.filename}:{self.lineno})"


class UsageStats:
    """Global usage statistics tracker.

    Automatically tracks token usage by model/provider and by caller.
    Thread-safe for concurrent access.
    """

    def __init__(self):
        """Initialize usage statistics tracker."""
        self._lock = Lock()
        self._total_by_model: defaultdict[str, int] = defaultdict(int)
        self._total_by_provider: defaultdict[str, int] = defaultdict(int)
        self._total_by_caller: defaultdict[CallerInfo, int] = defaultdict(int)
        self._detailed_records: list[dict] = []

    def record_usage(
        self,
        usage: TokenUsage,
        model: str,
        provider: str,
        caller_info: CallerInfo | None = None,
    ) -> None:
        """Record token usage statistics.

        Args:
            usage: Token usage information
            model: Model name (e.g., "gemini-2.5-flash", "gpt-4o")
            provider: Provider name ("gemini" or "openai")
            caller_info: Information about the caller (auto-detected if None)
        """
        if caller_info is None:
            caller_info = self._get_caller_info()

        with self._lock:
            self._total_by_model[model] += usage.total_tokens
            self._total_by_provider[provider] += usage.total_tokens
            self._total_by_caller[caller_info] += usage.total_tokens
            self._detailed_records.append(
                {
                    "model": model,
                    "provider": provider,
                    "caller": caller_info,
                    "usage": usage,
                }
            )

    def _get_caller_info(self, skip_frames: int = 3) -> CallerInfo:
        """Extract caller information from the call stack.

        Args:
            skip_frames: Number of frames to skip (default 3 to skip:
                         1. _get_caller_info itself
                         2. record_usage
                         3. _extract_*_usage function)
        """
        stack = inspect.stack()
        # Skip: this function, record_usage, extract function, and the actual LLM call function
        if len(stack) > skip_frames:
            frame = stack[skip_frames]
            module = inspect.getmodule(frame.frame)
            module_name = module.__name__ if module else None
            return CallerInfo(
                filename=frame.filename,
                function=frame.function,
                lineno=frame.lineno,
                module=module_name,
            )
        # Fallback if call stack is too short
        return CallerInfo(filename="unknown", function="unknown", lineno=0)

    def get_total_by_model(self) -> dict[str, int]:
        """Get total token usage per model."""
        with self._lock:
            return dict(self._total_by_model)

    def get_total_by_provider(self) -> dict[str, int]:
        """Get total token usage per provider."""
        with self._lock:
            return dict(self._total_by_provider)

    def get_total_by_caller(self) -> dict[str, int]:
        """Get total token usage per caller (as string representation)."""
        with self._lock:
            return {
                str(caller): tokens for caller, tokens in self._total_by_caller.items()
            }

    def get_detailed_records(self) -> list[dict]:
        """Get detailed records of all usage (copy of internal list)."""
        with self._lock:
            return self._detailed_records.copy()

    def get_summary(self) -> dict:
        """Get a summary of all usage statistics."""
        with self._lock:
            total = sum(self._total_by_model.values())
            return {
                "total_tokens": total,
                "by_model": dict(self._total_by_model),
                "by_provider": dict(self._total_by_provider),
                "by_caller": {
                    str(caller): tokens
                    for caller, tokens in self._total_by_caller.items()
                },
                "num_calls": len(self._detailed_records),
            }

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._total_by_model.clear()
            self._total_by_provider.clear()
            self._total_by_caller.clear()
            self._detailed_records.clear()

    def print_summary(self):
        summary = self.get_summary()
        print(f"\nTotal tokens used: {summary['total_tokens']}")
        print(f"Number of calls: {summary['num_calls']}")

        print("\n--- By Provider ---")
        for provider, tokens in summary["by_provider"].items():
            print(f"  {provider}: {tokens} tokens")

        print("\n--- By Model ---")
        for model, tokens in summary["by_model"].items():
            print(f"  {model}: {tokens} tokens")

        print("\n--- By Caller ---")
        for caller, tokens in sorted(
            summary["by_caller"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {tokens} tokens: {caller}")

        print("\n--- Detailed Records ---")
        for i, record in enumerate(usage_stats.get_detailed_records(), 1):
            print(f"\n  Call {i}:")
            print(f"    Model: {record['model']}")
            print(f"    Provider: {record['provider']}")
            print(f"    Caller: {record['caller']}")
            print(f"    Usage: {record['usage'].total_tokens} total tokens")
            print(
                f"      ({record['usage'].prompt_tokens} prompt + {record['usage'].completion_tokens} completion)"
            )


# Global stats instance
usage_stats = UsageStats()
