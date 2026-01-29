"""Tests for exceptions.py - provider requirement checks."""

from unittest.mock import patch

import pytest

from covenance.exceptions import MissingProviderError, require_provider


class TestRequireProvider:
    """Test require_provider import checks."""

    def test_openai_available_passes(self):
        """When openai is installed, require_provider succeeds."""
        # This should not raise since openai is in our test deps
        require_provider("openai")

    def test_anthropic_available_passes(self):
        """When anthropic is installed, require_provider succeeds."""
        require_provider("anthropic")

    def test_google_available_passes(self):
        """When google-genai is installed, require_provider succeeds."""
        require_provider("google")

    def test_mistral_available_passes(self):
        """When mistralai is installed, require_provider succeeds."""
        require_provider("mistral")

    def test_missing_openai_raises_helpful_error(self):
        """Missing openai package raises MissingProviderError with install hint."""
        with patch.dict("sys.modules", {"openai": None}):
            # Force ImportError by removing from modules
            import sys

            original = sys.modules.get("openai")
            sys.modules["openai"] = None  # type: ignore

            try:
                # Now the import inside require_provider should fail
                # We need to actually make import raise
                with patch(
                    "builtins.__import__",
                    side_effect=lambda name, *args: (_ for _ in ()).throw(ImportError())
                    if name == "openai"
                    else __builtins__.__dict__["__import__"](name, *args),
                ):
                    with pytest.raises(MissingProviderError) as exc_info:
                        require_provider("openai")
                    assert "pip install covenance[openai]" in str(exc_info.value)
            finally:
                if original:
                    sys.modules["openai"] = original

    def test_error_message_format(self):
        """Error message mentions package name and install command."""
        # Test with a mock import failure
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: (_ for _ in ()).throw(ImportError())
            if name == "anthropic"
            else __builtins__.__dict__["__import__"](name, *args),
        ):
            with pytest.raises(MissingProviderError) as exc_info:
                require_provider("anthropic")
            err_msg = str(exc_info.value)
            assert "anthropic" in err_msg
            assert "pip install" in err_msg


class TestExceptionTypes:
    """Test exception class definitions."""

    def test_missing_provider_error_is_import_error(self):
        """MissingProviderError subclasses ImportError for compatibility."""
        err = MissingProviderError("test")
        assert isinstance(err, ImportError)
