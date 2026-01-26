"""Unified routing implementation for OpenAI, Gemini, Mistral, Anthropic, OpenRouter.

This module provides a single interface that routes to the appropriate API
based on model name:
- Gemini models start with "gemini"
- Mistral models start with "mistral", "ministral", or "codestral"
- Claude models start with "claude" → Anthropic API
- OpenRouter models contain "/" (format: provider/model-name)
- All others route to OpenAI

Public API is exposed via `covenance.instance` which binds instance context.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import TypeAdapter, ValidationError

from covenance.clients.anthropic_client import ask_anthropic
from covenance.clients.anthropic_client import (
    set_rate_limiter_verbose as set_anthropic_verbose,
)
from covenance.clients.google_client import ask_gemini
from covenance.clients.google_client import (
    set_rate_limiter_verbose as set_gemini_verbose,
)
from covenance.clients.mistral_client import ask_mistral
from covenance.clients.mistral_client import (
    set_rate_limiter_verbose as set_mistral_verbose,
)
from covenance.clients.openai_client import ask_openai
from covenance.clients.openai_client import (
    set_rate_limiter_verbose as set_openai_verbose,
)
from covenance.clients.openrouter_client import ask_openrouter

from .exceptions import StructuredOutputParsingError


def set_rate_limiter_verbose(verbose: bool) -> None:
    """Enable or disable verbose logging for all rate limiters."""
    set_anthropic_verbose(verbose)
    set_gemini_verbose(verbose)
    set_openai_verbose(verbose)
    set_mistral_verbose(verbose)


def ask_llm[T](
    user_msg: str,
    model: str,
    format: type[T] | None = None,  # pydantic model, Literal[…] or None
    sys_msg: str | None = None,
    *,
    max_parsing_retries: int = 2,
) -> T:
    """
    Unified interface for structured LLM calls.

    Routes to the appropriate API based on model name:
    - Models starting with "gemini" → Google Gemini API
    - Models starting with "mistral", "ministral", or "codestral" → Mistral AI API
    - Models starting with "claude" → Anthropic Claude API
    - Models containing "/" (e.g., "openai/gpt-4o") → OpenRouter API
    - All others → OpenAI API

    Args:
        user_msg: User message/prompt
        model: Model name (required)
        format: Type schema (Pydantic model, list[BaseModel], dict[str, ...], etc.).
                Use `str` or `None` for plain text output.
        sys_msg: Optional system message/instructions
        max_parsing_retries: Maximum number of retries for parsing errors (default: 2)

    Returns:
        Parsed output of type T, or str if format is str or None

    Note:
        - Gemini supports more complex types like list[BaseModel] and dict[str, BaseModel]
        - Mistral supports Pydantic models via native structured output
        - Anthropic Claude supports Pydantic models via tools parameter with JSON schema
        - OpenAI supports Pydantic models and basic types
        - OpenRouter supports Pydantic models via OpenAI-compatible API
        - Token usage is always recorded in global stats (see usage_stats)
        - Automatically retries on StructuredOutputParsingError up to max_parsing_retries times
    """

    max_attempts = max_parsing_retries + 1  # Total attempts = retries + 1 initial attempt

    # Route based on model name
    if model.startswith("gemini"):
        llm_fn = ask_gemini
    elif model.startswith(("mistral", "ministral", "codestral")):
        llm_fn = ask_mistral
    elif model.startswith("claude"):
        llm_fn = ask_anthropic
    elif "/" in model:
        # OpenRouter models use format: provider/model-name
        llm_fn = ask_openrouter
    else:
        llm_fn = ask_openai

    for attempt in range(max_attempts):
        try:
            result = llm_fn(
                user_msg=user_msg,
                format=format,
                sys_msg=sys_msg,
                model=model,
            )
            if format not in (None, str):
                try:
                    result = TypeAdapter(format).validate_python(result)
                except ValidationError as exc:
                    raise StructuredOutputParsingError(
                        "Structured LLM output did not match expected schema."
                    ) from exc
            return result
        except StructuredOutputParsingError:
            if attempt == max_attempts - 1:
                raise
            # Retry on parsing errors
            continue


def llm_consensus[T](
    user_msg: str,
    model: str,
    format: type[T]
    | None = None,  # pydantic model, typing annotation, Literal[…], str, or None
    sys_msg: str | None = None,
    *,
    num_candidates: int = 3,
    additional_models: list[str] | None = None,
    integration_model: str | None = None,
    parallel: bool = True,
) -> T:
    """
    Make multiple LLM calls and integrate the results using an orchestrator LLM.

    This function makes multiple parallel calls to `ask_llm` with the same
    parameters, then uses an LLM orchestrator to integrate the candidate answers into
    a final result.

    Args:
        user_msg: User message/prompt
        model: Model name for candidate generation (required)
        format: Type schema (Pydantic model, list[BaseModel], dict[str, ...], etc.).
                Use `str` or `None` for plain text output.
        sys_msg: Optional system message/instructions
        num_candidates: Number of parallel LLM calls to make (default: 3).
            If num_candidates=1, makes a single direct call (no integration step).
        additional_models: List of models to cycle through for worker calls. If provided,
            worker calls will cycle through these models (e.g., [model1, model2, model3]
            for 5 calls would use: model1, model2, model3, model1, model2).
            If None, uses `model` for all worker calls.
        integration_model: Model name for integration call (defaults to same as model)
        parallel: Whether to make candidate calls in parallel (default: True)

    Returns:
        Parsed output of type T after integration, or str if format is str or None

    Note:
        - Individual candidate calls use the same parameters as a single call would
        - If `additional_models` is provided, worker calls cycle through those models
        - Integration call uses an orchestrator system prompt that instructs the LLM
          to follow the same rules as worker LLMs, plus the original system message
        - Integration call receives the original user message plus formatted candidate answers
    """

    # Short-circuit: num_candidates=1 means direct call, no consensus
    if num_candidates == 1:
        return ask_llm(
            user_msg=user_msg,
            format=format,
            sys_msg=sys_msg,
            model=model,
        )

    if integration_model is None:
        integration_model = model

    # Determine which models to use for worker calls
    if additional_models and len(additional_models) > 0:
        worker_models = additional_models
    else:
        worker_models = [model]

    # Get current LLM operation context to propagate to worker threads
    # Context variables don't automatically propagate to ThreadPoolExecutor threads
    from .metrics import LLMOperationContext
    from .client_context import snapshot as snapshot_instance_context
    from .client_context import restore as restore_instance_context

    current_context = LLMOperationContext.current()
    current_instance_context = snapshot_instance_context()

    # Make multiple candidate calls
    def make_candidate_call(call_index: int) -> T:
        # Propagate context to this thread
        LLMOperationContext.set_current(current_context)
        restore_instance_context(current_instance_context)

        # Cycle through worker models
        worker_model = worker_models[call_index % len(worker_models)]
        return ask_llm(
            user_msg=user_msg,
            format=format,
            sys_msg=sys_msg,
            model=worker_model,
        )

    candidates: list[T] = []
    if parallel:
        # Make parallel calls with context propagation
        with ThreadPoolExecutor(max_workers=num_candidates) as executor:
            futures = [
                executor.submit(make_candidate_call, i) for i in range(num_candidates)
            ]
            for future in as_completed(futures):
                try:
                    candidates.append(future.result())
                except Exception as e:
                    # If a call fails, we continue with the successful ones
                    # This follows the principle of not hedging unnecessarily
                    raise RuntimeError(
                        f"Failed to generate candidate answer: {e}"
                    ) from e
    else:
        # Make sequential calls
        for i in range(num_candidates):
            candidates.append(make_candidate_call(i))

    # Format candidate answers for integration
    candidate_texts = []
    for i, candidate in enumerate(candidates, 1):
        if hasattr(candidate, "model_dump"):
            # Pydantic model
            candidate_json = json.dumps(
                candidate.model_dump(mode="json"), ensure_ascii=False, indent=2
            )
        elif isinstance(candidate, (dict, list)):
            # Dict or list
            candidate_json = json.dumps(candidate, ensure_ascii=False, indent=2)
        else:
            # Other types - convert to string
            candidate_json = str(candidate)

        candidate_texts.append(f"--- Candidate Answer {i} ---\n{candidate_json}")

    # Create integration prompt
    integration_user_msg = f"""{user_msg}

Below are {len(candidates)} candidate answers generated by worker LLMs. Please integrate them into a single, high-quality answer that follows the same format and requirements as specified above.

"""
    # Add all candidates
    for candidate_text in candidate_texts:
        integration_user_msg += f"\n{candidate_text}\n"

    # Create integration system message
    integration_sys_msg = (
        "You are an LLM orchestrator, your goal is to integrate individual answers into a high quality answer. "
        f"Worker system message: {sys_msg or 'you are a helpful assistant'}"
    )

    # Make integration call
    return ask_llm(
        user_msg=integration_user_msg,
        format=format,
        sys_msg=integration_sys_msg,
        model=integration_model,
    )
