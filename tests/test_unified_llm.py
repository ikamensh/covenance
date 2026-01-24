"""Tests for the unified LLM wrapper.

These tests specifically test the LLM wrapper functions themselves,
so they need the real implementations (with mocked underlying providers).
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

# Import the covenance module - tests use covenance.ask_llm_structured directly
import covenance
from covenance.exceptions import StructuredOutputParsingError
from covenance.usage import usage_stats


class SimpleResponse(BaseModel):
    """Simple response model for testing."""

    answer: str
    confidence: float


@pytest.fixture(autouse=True)
def unblock_llm_for_module(unblock_llm):
    """Apply unblock_llm to all tests in this module.

    The unblock_llm fixture has already restored the functions in covenance module.
    Tests use covenance.ask_llm_structured directly.
    """
    yield


@pytest.fixture(autouse=True)
def reset_usage_stats():
    """Reset usage stats before each test."""
    usage_stats.reset()
    yield
    usage_stats.reset()


def test_routes_to_gemini_when_model_starts_with_gemini():
    """Form test: verifies models starting with 'gemini' route to Gemini API with correct structure."""
    mock_response = MagicMock()
    mock_response.parsed = SimpleResponse(answer="Paris", confidence=0.95)
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    with patch("covenance.google_client.client.models.generate_content") as mock_gemini:
        mock_gemini.return_value = mock_response

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            sys_msg="Answer concisely.",
            model="gemini-2.5-flash",
        )

    assert isinstance(result, SimpleResponse)
    assert result.answer == "Paris"
    assert result.confidence == 0.95

    mock_gemini.assert_called_once()
    call_kwargs = mock_gemini.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["contents"] == "What is the capital of France?"
    assert call_kwargs["config"]["response_mime_type"] == "application/json"
    assert call_kwargs["config"]["response_schema"] == SimpleResponse
    assert call_kwargs["config"]["system_instruction"] == "Answer concisely."


def test_routes_to_openai_when_model_does_not_start_with_gemini():
    """Form test: verifies non-gemini models route to OpenAI API with correct structure."""
    mock_response = MagicMock()
    mock_response.output_parsed = SimpleResponse(answer="Paris", confidence=0.95)
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.total_tokens = 15

    with patch("covenance.openai_client.client.responses.parse") as mock_openai:
        mock_openai.return_value = mock_response

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            sys_msg="Answer concisely.",
            model="gpt-4o",
        )

    assert isinstance(result, SimpleResponse)
    assert result.answer == "Paris"
    assert result.confidence == 0.95

    mock_openai.assert_called_once()
    call_kwargs = mock_openai.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["input"] == "What is the capital of France?"
    assert call_kwargs["text_format"] == SimpleResponse
    assert call_kwargs["instructions"] == "Answer concisely."


def test_gemini_without_system_message():
    """Form test: verifies Gemini API handles optional system message parameter correctly."""
    mock_response = MagicMock()
    mock_response.parsed = SimpleResponse(answer="Paris", confidence=0.95)
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    with patch("covenance.google_client.client.models.generate_content") as mock_gemini:
        mock_gemini.return_value = mock_response

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            model="gemini-2.5-flash",
        )

    assert isinstance(result, SimpleResponse)
    call_kwargs = mock_gemini.call_args.kwargs
    assert "system_instruction" not in call_kwargs["config"]


def test_openai_without_system_message():
    """Form test: verifies OpenAI API handles optional system message parameter correctly."""
    mock_response = MagicMock()
    mock_response.output_parsed = SimpleResponse(answer="Paris", confidence=0.95)
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.total_tokens = 15

    with patch("covenance.openai_client.client.responses.parse") as mock_openai:
        mock_openai.return_value = mock_response

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            model="gpt-4o",
        )

    assert isinstance(result, SimpleResponse)
    call_kwargs = mock_openai.call_args.kwargs
    assert call_kwargs["instructions"] is None


def test_usage_stats_recorded_for_gemini():
    """Property test: verifies usage statistics are recorded correctly for Gemini calls."""
    mock_response = MagicMock()
    mock_response.parsed = SimpleResponse(answer="Paris", confidence=0.95)
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    with patch("covenance.google_client.client.models.generate_content") as mock_gemini:
        mock_gemini.return_value = mock_response

        initial_calls = len(usage_stats.get_detailed_records())

        covenance.ask_llm(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            model="gemini-2.5-flash",
        )

    records = usage_stats.get_detailed_records()
    assert len(records) == initial_calls + 1

    last_record = records[-1]
    assert last_record["model"] == "gemini-2.5-flash"
    assert last_record["provider"] == "gemini"
    assert last_record["usage"].total_tokens == 15
    assert last_record["usage"].prompt_tokens == 10
    assert last_record["usage"].completion_tokens == 5

    summary = usage_stats.get_summary()
    assert summary["by_model"]["gemini-2.5-flash"] == 15
    assert summary["by_provider"]["gemini"] == 15


def test_usage_stats_recorded_for_openai():
    """Property test: verifies usage statistics are recorded correctly for OpenAI calls."""
    mock_response = MagicMock()
    mock_response.output_parsed = SimpleResponse(answer="Paris", confidence=0.95)
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 20
    mock_response.usage.output_tokens = 10
    mock_response.usage.total_tokens = 30

    with patch("covenance.openai_client.client.responses.parse") as mock_openai:
        mock_openai.return_value = mock_response

        initial_calls = len(usage_stats.get_detailed_records())

        covenance.ask_llm(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            model="gpt-4o",
        )

    records = usage_stats.get_detailed_records()
    assert len(records) == initial_calls + 1

    last_record = records[-1]
    assert last_record["model"] == "gpt-4o"
    assert last_record["provider"] == "openai"
    assert last_record["usage"].total_tokens == 30
    assert last_record["usage"].prompt_tokens == 20
    assert last_record["usage"].completion_tokens == 10

    summary = usage_stats.get_summary()
    assert summary["by_model"]["gpt-4o"] == 30
    assert summary["by_provider"]["openai"] == 30


def test_multiple_calls_accumulate_stats():
    """Property test: verifies usage statistics accumulate correctly across multiple calls and providers."""
    mock_gemini_response = MagicMock()
    mock_gemini_response.parsed = SimpleResponse(answer="Paris", confidence=0.95)
    mock_gemini_response.usage_metadata = MagicMock()
    mock_gemini_response.usage_metadata.prompt_token_count = 10
    mock_gemini_response.usage_metadata.candidates_token_count = 5
    mock_gemini_response.usage_metadata.total_token_count = 15

    mock_openai_response = MagicMock()
    mock_openai_response.output_parsed = SimpleResponse(
        answer="London", confidence=0.90
    )
    mock_openai_response.usage = MagicMock()
    mock_openai_response.usage.input_tokens = 20
    mock_openai_response.usage.output_tokens = 10
    mock_openai_response.usage.total_tokens = 30

    with (
        patch("covenance.google_client.client.models.generate_content") as mock_gemini,
        patch("covenance.openai_client.client.responses.parse") as mock_openai,
    ):
        mock_gemini.return_value = mock_gemini_response
        mock_openai.return_value = mock_openai_response

        covenance.ask_llm(
            user_msg="Question 1",
            format=SimpleResponse,
            model="gemini-2.5-flash",
        )

        covenance.ask_llm(
            user_msg="Question 2",
            format=SimpleResponse,
            model="gpt-4o",
        )

        covenance.ask_llm(
            user_msg="Question 3",
            format=SimpleResponse,
            model="gemini-2.5-flash",
        )

    summary = usage_stats.get_summary()
    assert summary["total_tokens"] == 60  # 15 + 30 + 15
    assert summary["by_provider"]["gemini"] == 30  # 15 + 15
    assert summary["by_provider"]["openai"] == 30
    assert summary["by_model"]["gemini-2.5-flash"] == 30
    assert summary["by_model"]["gpt-4o"] == 30
    assert summary["num_calls"] == 3


def test_openai_usage_extraction_fallback():
    """Form test: verifies OpenAI usage extraction works with usage object."""
    mock_response = MagicMock()
    mock_response.output_parsed = SimpleResponse(answer="Paris", confidence=0.95)
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 25
    mock_response.usage.output_tokens = 15
    mock_response.usage.total_tokens = 40

    with patch("covenance.openai_client.client.responses.parse") as mock_openai:
        mock_openai.return_value = mock_response

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            model="gpt-4o",
        )

    assert isinstance(result, SimpleResponse)
    records = usage_stats.get_detailed_records()
    assert records[-1]["usage"].total_tokens == 40


def test_openai_usage_extraction_raises_when_missing():
    """Form test: verifies OpenAI usage extraction raises exception when usage information is missing."""

    # Create a mock response without usage attributes using spec
    class ResponseSpec:
        output_parsed = None

    mock_response = MagicMock(spec=ResponseSpec)
    mock_response.output_parsed = SimpleResponse(answer="Paris", confidence=0.95)
    # usage attribute won't exist due to spec, so hasattr will return False

    with patch("covenance.openai_client.client.responses.parse") as mock_openai:
        mock_openai.return_value = mock_response

        with pytest.raises(AttributeError, match="OpenAI response missing usage info"):
            covenance.ask_llm(
                user_msg="What is the capital of France?",
                format=SimpleResponse,
                model="gpt-4o",
            )


def test_ask_llm_structured_with_consensus_makes_multiple_calls():
    """Form test: verifies consensus function makes multiple candidate calls and integration call."""
    # Mock responses for candidate calls
    candidate_responses = [
        SimpleResponse(answer="Paris", confidence=0.95),
        SimpleResponse(answer="Paris", confidence=0.90),
        SimpleResponse(answer="Lyon", confidence=0.85),
    ]

    # Mock response for integration call
    integration_response = SimpleResponse(answer="Paris", confidence=0.92)

    call_count = {"candidate": 0, "integration": 0}
    candidate_index = 0

    def mock_ask_llm(*args, **kwargs):
        nonlocal candidate_index
        # Check if this is an integration call by looking at user_msg
        user_msg = kwargs.get("user_msg", args[0] if args else "")
        # Integration call contains "candidate answers" text
        if (
            "candidate answers" in user_msg.lower()
            and "worker llms" in user_msg.lower()
        ):
            call_count["integration"] += 1
            return integration_response
        else:
            call_count["candidate"] += 1
            result = candidate_responses[candidate_index % len(candidate_responses)]
            candidate_index += 1
            return result

    with patch("covenance.unified.ask_llm_structured") as mock_ask:
        mock_ask.side_effect = mock_ask_llm

        result = covenance.llm_consensus(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            sys_msg="Answer concisely.",
            model="gemini-2.5-flash",
            num_candidates=3,
            parallel=False,  # Use sequential to avoid race conditions in test
        )

    assert isinstance(result, SimpleResponse)
    assert result.answer == "Paris"
    assert result.confidence == 0.92
    assert call_count["candidate"] == 3
    assert call_count["integration"] == 1
    assert mock_ask.call_count == 4


def test_ask_llm_structured_with_consensus_integration_prompt():
    """Form test: verifies integration call receives correct system and user messages."""
    candidate_response = SimpleResponse(answer="Paris", confidence=0.95)
    integration_response = SimpleResponse(answer="Paris", confidence=0.92)

    integration_calls = []

    def mock_ask_llm(*args, **kwargs):
        # Capture integration call arguments
        user_msg = kwargs.get("user_msg", args[0] if args else "")
        # Integration call contains "candidate answers" text
        if (
            "candidate answers" in user_msg.lower()
            and "worker llms" in user_msg.lower()
        ):
            integration_calls.append(
                {
                    "user_msg": kwargs.get("user_msg", args[0] if args else ""),
                    "sys_msg": kwargs.get(
                        "sys_msg", args[2] if len(args) > 2 else None
                    ),
                }
            )
            return integration_response
        else:
            return candidate_response

    with patch("covenance.unified.ask_llm_structured") as mock_ask:
        mock_ask.side_effect = mock_ask_llm

        covenance.llm_consensus(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            sys_msg="Answer concisely.",
            model="gemini-2.5-flash",
            num_candidates=3,
            parallel=False,  # Use sequential to avoid race conditions in test
        )

    # Verify integration call
    assert len(integration_calls) == 1
    integration_call = integration_calls[0]

    # Check system message includes orchestrator instruction + original sys_msg
    assert "You are an LLM orchestrator" in integration_call["sys_msg"]
    assert "Answer concisely." in integration_call["sys_msg"]

    # Check user message includes original message and candidate answers
    user_content = integration_call["user_msg"]
    assert "What is the capital of France?" in user_content
    assert "candidate answers" in user_content.lower()
    assert "worker llms" in user_content.lower()
    assert "Candidate Answer 1" in user_content
    assert "Candidate Answer 2" in user_content
    assert "Candidate Answer 3" in user_content


def test_ask_llm_structured_with_consensus_cycles_models():
    """Form test: verifies consensus function cycles through additional_models for worker calls."""
    candidate_response = SimpleResponse(answer="Paris", confidence=0.95)
    integration_response = SimpleResponse(answer="Paris", confidence=0.92)

    worker_calls = []

    def mock_ask_llm(*args, **kwargs):
        # Capture worker call arguments
        user_msg = kwargs.get("user_msg", args[0] if args else "")
        # Integration call contains "candidate answers" text
        if (
            "candidate answers" in user_msg.lower()
            and "worker llms" in user_msg.lower()
        ):
            return integration_response
        else:
            # Capture worker call model
            worker_calls.append(
                {
                    "model": kwargs.get("model", args[3] if len(args) > 3 else None),
                }
            )
            return candidate_response

    with patch("covenance.unified.ask_llm_structured") as mock_ask:
        mock_ask.side_effect = mock_ask_llm

        covenance.llm_consensus(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            sys_msg="Answer concisely.",
            model="gemini-2.5-flash",  # Base model (not used when additional_models provided)
            num_candidates=5,
            additional_models=["model1", "model2", "model3"],
            parallel=False,  # Use sequential to verify order
        )

    # Verify worker calls cycled through models: model1, model2, model3, model1, model2
    assert len(worker_calls) == 5
    assert worker_calls[0]["model"] == "model1"
    assert worker_calls[1]["model"] == "model2"
    assert worker_calls[2]["model"] == "model3"
    assert worker_calls[3]["model"] == "model1"
    assert worker_calls[4]["model"] == "model2"


def test_ask_llm_structured_with_consensus_uses_base_model_when_no_additional():
    """Form test: verifies consensus function uses base model when additional_models not provided."""
    candidate_response = SimpleResponse(answer="Paris", confidence=0.95)
    integration_response = SimpleResponse(answer="Paris", confidence=0.92)

    worker_calls = []

    def mock_ask_llm(*args, **kwargs):
        # Capture worker call arguments
        user_msg = kwargs.get("user_msg", args[0] if args else "")
        # Integration call contains "candidate answers" text
        if (
            "candidate answers" in user_msg.lower()
            and "worker llms" in user_msg.lower()
        ):
            return integration_response
        else:
            # Capture worker call model
            worker_calls.append(
                {
                    "model": kwargs.get("model", args[3] if len(args) > 3 else None),
                }
            )
            return candidate_response

    with patch("covenance.unified.ask_llm_structured") as mock_ask:
        mock_ask.side_effect = mock_ask_llm

        covenance.llm_consensus(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            sys_msg="Answer concisely.",
            model="gemini-2.5-flash",
            num_candidates=3,
            additional_models=None,  # No additional models
            parallel=False,  # Use sequential to verify order
        )

    # Verify all worker calls used the base model
    assert len(worker_calls) == 3
    assert all(call["model"] == "gemini-2.5-flash" for call in worker_calls)


def test_ask_llm_structured_with_consensus_empty_additional_models_uses_base():
    """Form test: verifies consensus function uses base model when additional_models is empty list."""
    candidate_response = SimpleResponse(answer="Paris", confidence=0.95)
    integration_response = SimpleResponse(answer="Paris", confidence=0.92)

    worker_calls = []

    def mock_ask_llm(*args, **kwargs):
        # Capture worker call arguments
        user_msg = kwargs.get("user_msg", args[0] if args else "")
        # Integration call contains "candidate answers" text
        if (
            "candidate answers" in user_msg.lower()
            and "worker llms" in user_msg.lower()
        ):
            return integration_response
        else:
            # Capture worker call model
            worker_calls.append(
                {
                    "model": kwargs.get("model", args[3] if len(args) > 3 else None),
                }
            )
            return candidate_response

    with patch("covenance.unified.ask_llm_structured") as mock_ask:
        mock_ask.side_effect = mock_ask_llm

        covenance.llm_consensus(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            sys_msg="Answer concisely.",
            model="gemini-2.5-flash",
            num_candidates=3,
            additional_models=[],  # Empty list should fall back to base model
            parallel=False,  # Use sequential to verify order
        )

    # Verify all worker calls used the base model (empty list treated as None)
    assert len(worker_calls) == 3
    assert all(call["model"] == "gemini-2.5-flash" for call in worker_calls)


def test_retry_on_parsing_error():
    """Property test: verifies retry logic works correctly for parsing errors."""
    mock_response_success = MagicMock()
    mock_response_success.parsed = SimpleResponse(answer="Paris", confidence=0.95)
    mock_response_success.usage_metadata = MagicMock()
    mock_response_success.usage_metadata.prompt_token_count = 10
    mock_response_success.usage_metadata.candidates_token_count = 5
    mock_response_success.usage_metadata.total_token_count = 15

    mock_response_failure = MagicMock()
    mock_response_failure.parsed = None  # Simulate parsing error
    mock_response_failure.usage_metadata = MagicMock()
    mock_response_failure.usage_metadata.prompt_token_count = 10
    mock_response_failure.usage_metadata.candidates_token_count = 5
    mock_response_failure.usage_metadata.total_token_count = 15

    with patch("covenance.google_client.client.models.generate_content") as mock_gemini:
        # First call fails, second succeeds
        mock_gemini.side_effect = [mock_response_failure, mock_response_success]

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            format=SimpleResponse,
            model="gemini-2.5-flash",
            max_parsing_retries=2,
        )

    assert isinstance(result, SimpleResponse)
    assert result.answer == "Paris"
    assert mock_gemini.call_count == 2  # Should have retried once


def test_retry_exhausted_raises_exception():
    """Property test: verifies exception is raised when all retries are exhausted."""
    mock_response_failure = MagicMock()
    mock_response_failure.parsed = None  # Simulate parsing error
    mock_response_failure.usage_metadata = MagicMock()
    mock_response_failure.usage_metadata.prompt_token_count = 10
    mock_response_failure.usage_metadata.candidates_token_count = 5
    mock_response_failure.usage_metadata.total_token_count = 15

    with patch("covenance.google_client.client.models.generate_content") as mock_gemini:
        # All calls fail
        mock_gemini.return_value = mock_response_failure

        with pytest.raises(StructuredOutputParsingError):
            covenance.ask_llm(
                user_msg="What is the capital of France?",
                format=SimpleResponse,
                model="gemini-2.5-flash",
                max_parsing_retries=2,
            )

    # Should have tried 3 times (1 initial + 2 retries)
    assert mock_gemini.call_count == 3
