"""API key resolution helpers."""

from __future__ import annotations

import os
from collections.abc import Iterable

from dotenv import find_dotenv, load_dotenv

# Registry mapping provider names to their environment variable names
_API_KEY_ENV_VARS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "grok": ["XAI_API_KEY", "GROK_API_KEY"],
}


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def load_env_if_present() -> None:
    """Load environment variables from a .env file if it exists."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)


def _get_key(env_vars: Iterable[str]) -> str | None:
    load_env_if_present()
    return _first_env(*env_vars)


def require_api_key(
    key: str | None, provider: str, env_vars: Iterable[str] | None = None
) -> str:
    if key:
        return key
    if env_vars is None:
        env_vars = _API_KEY_ENV_VARS.get(provider, [])
    env_list = " or ".join(env_vars)
    raise RuntimeError(
        f"Missing {provider} API key. Please set {env_list} environment variable "
        "(you can also use a .env file)."
    )


def get_openai_api_key() -> str | None:
    return _get_key(_API_KEY_ENV_VARS["openai"])


def get_anthropic_api_key() -> str | None:
    return _get_key(_API_KEY_ENV_VARS["anthropic"])


def get_mistral_api_key() -> str | None:
    return _get_key(_API_KEY_ENV_VARS["mistral"])


def get_openrouter_api_key() -> str | None:
    return _get_key(_API_KEY_ENV_VARS["openrouter"])


def get_gemini_api_key() -> str | None:
    return _get_key(_API_KEY_ENV_VARS["gemini"])


def get_grok_api_key() -> str | None:
    return _get_key(_API_KEY_ENV_VARS["grok"])
