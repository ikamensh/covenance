"""Model name types for LLM API calls.

This module defines a Literal type that includes all valid model names
from Gemini, OpenAI, Mistral, Anthropic Claude, and OpenRouter, which can be used in Pydantic models to
generate proper OpenAPI schemas with enum values.

The Literal type is derived dynamically from the enums to ensure it stays
in sync when new models are added.
"""

from typing import Literal

from .anthropic_client import ClaudeModels
from .google_client import GeminiModels
from .mistral_client import MistralModels
from .openai_client import OpenaiModels as OpenAIModels
from .openrouter_client import OpenRouterModels


def _create_llm_model_name_literal() -> type:
    """Create a Literal type from all model enum values.

    Dynamically extracts all model names from GeminiModels, OpenAIModels,
    MistralModels, ClaudeModels, and OpenRouterModels enums and creates a Literal type that Pydantic can
    introspect for OpenAPI schema generation.

    Returns:
        A Literal type containing all valid model names
    """
    # Collect all model values from all enums
    gemini_models = [model.value for model in GeminiModels]
    openai_models = [model.value for model in OpenAIModels]
    mistral_models = [model.value for model in MistralModels]
    claude_models = [model.value for model in ClaudeModels]
    openrouter_models = [model.value for model in OpenRouterModels]
    all_models = tuple(
        gemini_models
        + openai_models
        + mistral_models
        + claude_models
        + openrouter_models
    )

    # Create Literal type dynamically by unpacking the tuple
    # In Python 3.11+, we can use Literal[*values] syntax
    return Literal[*all_models]  # type: ignore[misc]


# Export the Literal type for use in Pydantic models
# This is created dynamically from the enums to stay in sync
LLMModelName = _create_llm_model_name_literal()

# Default model (Gemini flash)
DEFAULT_MODEL: LLMModelName = GeminiModels.flash3
