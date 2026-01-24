"""Tests for .env loading behavior.

Doctest summary of expected resolution order:
>>> def resolve(env, dotenv):
...     return env.get("OPENAI_API_KEY") or dotenv.get("OPENAI_API_KEY")
>>> resolve({}, {"OPENAI_API_KEY": "from-dotenv"})
'from-dotenv'
>>> resolve({"OPENAI_API_KEY": "from-env"}, {"OPENAI_API_KEY": "from-dotenv"})
'from-env'
"""

import covenance.keys as config


def test_loads_dotenv_when_env_missing(monkeypatch, tmp_path):
    """Loads a .env value when the environment variable is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert config.get_openai_api_key() == "from-dotenv"


def test_dotenv_does_not_override_existing_env(monkeypatch, tmp_path):
    """Keeps existing environment variables even when .env defines a value."""
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert config.get_openai_api_key() == "from-env"
