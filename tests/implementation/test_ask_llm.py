"""Implementation tests for ask_llm and llm_consensus.

Tests internal behavior by mocking backend calls:
- Native backend routing (OpenAI, Grok)
- Pydantic-AI backend routing (Gemini, Anthropic, Mistral)
- Consensus orchestration
- Retry logic

These tests depend on internal implementation details.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import covenance
from covenance._backend_result import BackendResult
from covenance.exceptions import StructuredOutputParsingError
from covenance.record import TokenUsage


class SimpleResponse(BaseModel):
    """Simple response model for testing."""

    answer: str
    confidence: float


def _mock_usage() -> TokenUsage:
    """Create a mock TokenUsage for testing."""
    return TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def _mock_backend_result(output) -> BackendResult:
    """Create a mock BackendResult for testing."""
    return BackendResult(output=output, usage=_mock_usage())


@pytest.fixture(autouse=True)
def unblock_llm_for_module(unblock_llm):
    """Apply unblock_llm to all tests in this module."""
    yield


def test_routes_to_pydantic_ai_backend_for_gemini():
    """Gemini models use pydantic-ai backend."""
    expected = SimpleResponse(answer="Paris", confidence=0.95)

    with patch("covenance.client.ask_pydantic_ai") as mock_backend:
        mock_backend.return_value = _mock_backend_result(expected)

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            model="gemini-2.5-flash",
        )

    assert result == expected
    mock_backend.assert_called_once()
    call_kwargs = mock_backend.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"


def test_routes_to_native_backend_for_openai():
    """OpenAI models use native backend."""
    expected = SimpleResponse(answer="Paris", confidence=0.95)

    with patch(
        "covenance.clients.openai_client.ask_openai_compatible_structured"
    ) as mock_native:
        mock_native.return_value = _mock_backend_result(expected)

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            model="gpt-4o",
        )

    assert result == expected
    mock_native.assert_called_once()
    call_kwargs = mock_native.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["provider"] == "openai"


def test_routes_to_native_backend_for_grok():
    """Grok models use native backend."""
    expected = SimpleResponse(answer="Paris", confidence=0.95)

    with patch(
        "covenance.clients.openai_client.ask_openai_compatible_structured"
    ) as mock_native:
        mock_native.return_value = _mock_backend_result(expected)

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            model="grok-4",
        )

    assert result == expected
    mock_native.assert_called_once()
    call_kwargs = mock_native.call_args.kwargs
    assert call_kwargs["model"] == "grok-4"
    assert call_kwargs["provider"] == "grok"


def test_routes_to_pydantic_ai_backend_for_anthropic():
    """Anthropic models use pydantic-ai backend."""
    expected = SimpleResponse(answer="Paris", confidence=0.95)

    with patch("covenance.client.ask_pydantic_ai") as mock_backend:
        mock_backend.return_value = _mock_backend_result(expected)

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            model="claude-3-sonnet",
        )

    assert result == expected
    mock_backend.assert_called_once()


def test_routes_to_pydantic_ai_backend_for_mistral():
    """Mistral models use pydantic-ai backend."""
    expected = SimpleResponse(answer="Paris", confidence=0.95)

    with patch("covenance.client.ask_pydantic_ai") as mock_backend:
        mock_backend.return_value = _mock_backend_result(expected)

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            model="mistral-large",
        )

    assert result == expected
    mock_backend.assert_called_once()


def test_consensus_makes_multiple_calls():
    """Consensus function makes multiple candidate calls and integration call."""
    candidate_responses = [
        SimpleResponse(answer="Paris", confidence=0.95),
        SimpleResponse(answer="Paris", confidence=0.90),
        SimpleResponse(answer="Lyon", confidence=0.85),
    ]
    integration_response = SimpleResponse(answer="Paris", confidence=0.92)

    call_count = {"candidate": 0, "integration": 0}
    candidate_index = 0

    def mock_ask_pydantic_ai(*args, **kwargs):
        nonlocal candidate_index
        user_msg = kwargs.get("user_msg", "")
        if "candidate answers" in user_msg.lower():
            call_count["integration"] += 1
            result = integration_response
        else:
            call_count["candidate"] += 1
            result = candidate_responses[candidate_index % len(candidate_responses)]
            candidate_index += 1
        return _mock_backend_result(result)

    with patch("covenance.client.ask_pydantic_ai") as mock_backend:
        mock_backend.side_effect = mock_ask_pydantic_ai

        result = covenance.llm_consensus(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            model="gemini-2.5-flash",
            num_candidates=3,
            parallel=False,
        )

    assert result == integration_response
    assert call_count["candidate"] == 3
    assert call_count["integration"] == 1


def test_consensus_integration_prompt_structure():
    """Integration call receives correct system and user messages."""
    candidate_response = SimpleResponse(answer="Paris", confidence=0.95)
    integration_response = SimpleResponse(answer="Paris", confidence=0.92)

    integration_calls = []

    def mock_ask_pydantic_ai(*args, **kwargs):
        user_msg = kwargs.get("user_msg", "")
        if "candidate answers" in user_msg.lower():
            integration_calls.append(kwargs)
            return _mock_backend_result(integration_response)
        return _mock_backend_result(candidate_response)

    with patch("covenance.client.ask_pydantic_ai") as mock_backend:
        mock_backend.side_effect = mock_ask_pydantic_ai

        covenance.llm_consensus(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            sys_msg="Answer concisely.",
            model="gemini-2.5-flash",
            num_candidates=3,
            parallel=False,
        )

    assert len(integration_calls) == 1
    integration_call = integration_calls[0]

    # Check system message includes orchestrator instruction + original sys_msg
    assert "You are an LLM orchestrator" in integration_call["sys_msg"]
    assert "Answer concisely." in integration_call["sys_msg"]

    # Check user message includes original message and candidate answers
    user_content = integration_call["user_msg"]
    assert "What is the capital of France?" in user_content
    assert "Candidate Answer 1" in user_content
    assert "Candidate Answer 2" in user_content
    assert "Candidate Answer 3" in user_content


def test_retry_on_parsing_error_native_backend():
    """Retry logic works for parsing errors on native backend."""
    success_result = _mock_backend_result(
        SimpleResponse(answer="Paris", confidence=0.95)
    )

    fail_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    attempt = {"count": 0}

    def mock_native(*args, **kwargs):
        attempt["count"] += 1
        if attempt["count"] == 1:
            raise StructuredOutputParsingError("Parse failed", usage=fail_usage)
        return success_result

    with patch(
        "covenance.clients.openai_client.ask_openai_compatible_structured"
    ) as mock:
        mock.side_effect = mock_native

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            model="gpt-4o",
            max_parsing_retries=2,
        )

    assert result.answer == "Paris"
    assert attempt["count"] == 2


def test_retry_exhausted_raises_exception():
    """Exception is raised when all retries are exhausted."""
    fail_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    def mock_native(*args, **kwargs):
        raise StructuredOutputParsingError("Parse failed", usage=fail_usage)

    with patch(
        "covenance.clients.openai_client.ask_openai_compatible_structured"
    ) as mock:
        mock.side_effect = mock_native

        with pytest.raises(StructuredOutputParsingError):
            covenance.ask_llm(
                user_msg="What is the capital of France?",
                response_type=SimpleResponse,
                model="gpt-4o",
                max_parsing_retries=2,
            )

    # Should have tried 3 times (1 initial + 2 retries)
    assert mock.call_count == 3


def test_records_consolidated_after_retries():
    """Token usage from all retry attempts is consolidated into single record."""
    usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    usage2 = TokenUsage(prompt_tokens=100, completion_tokens=60, total_tokens=160)
    usage3 = TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140)

    attempt = {"count": 0}

    def mock_native(*args, **kwargs):
        attempt["count"] += 1
        if attempt["count"] <= 2:
            raise StructuredOutputParsingError(
                "Parse failed",
                usage=usage1 if attempt["count"] == 1 else usage2,
            )
        return BackendResult(
            output=SimpleResponse(answer="Paris", confidence=0.95),
            usage=usage3,
        )

    covenance.clear_records()

    with patch(
        "covenance.clients.openai_client.ask_openai_compatible_structured"
    ) as mock:
        mock.side_effect = mock_native

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            model="gpt-4o",
            max_parsing_retries=2,
        )

    assert result.answer == "Paris"

    # Single consolidated record
    records = covenance.get_records()
    assert len(records) == 1

    record = records[0]
    assert record.structured_output_retries == 2
    assert record.tokens_input == 300  # 100 + 100 + 100
    assert record.tokens_output == 150  # 50 + 60 + 40
