"""Implementation tests for ask_llm and llm_consensus.

Tests internal behavior by mocking underlying provider calls:
- Request/response structure passed to providers
- Consensus orchestration (candidate calls, integration)
- Retry logic and token aggregation

These tests depend on internal implementation details.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

# Import the covenance module - tests use covenance.ask_llm directly
import covenance
from covenance.exceptions import StructuredOutputParsingError
from covenance.record import RawCallResult, TokenUsage


class SimpleResponse(BaseModel):
    """Simple response model for testing."""

    answer: str
    confidence: float


def _mock_usage() -> TokenUsage:
    """Create a mock TokenUsage for testing."""
    return TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


@pytest.fixture(autouse=True)
def unblock_llm_for_module(unblock_llm):
    """Apply unblock_llm to all tests in this module.

    The unblock_llm fixture has already restored the functions in covenance module.
    Tests use covenance.ask_llm directly.
    """
    yield


def test_routes_to_gemini_when_model_starts_with_gemini():
    """Form test: verifies models starting with 'gemini' route to Gemini API with correct structure."""
    mock_response = MagicMock()
    mock_response.parsed = SimpleResponse(answer="Paris", confidence=0.95)
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    with patch(
        "covenance.clients.google_client.client.models.generate_content"
    ) as mock_gemini:
        mock_gemini.return_value = mock_response

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
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

    with patch("covenance.clients.openai_client.client.responses.parse") as mock_openai:
        mock_openai.return_value = mock_response

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
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

    with patch(
        "covenance.clients.google_client.client.models.generate_content"
    ) as mock_gemini:
        mock_gemini.return_value = mock_response

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
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

    with patch("covenance.clients.openai_client.client.responses.parse") as mock_openai:
        mock_openai.return_value = mock_response

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            model="gpt-4o",
        )

    assert isinstance(result, SimpleResponse)
    call_kwargs = mock_openai.call_args.kwargs
    assert call_kwargs["instructions"] is None


def test_openai_usage_extraction_fallback():
    """Form test: verifies OpenAI usage extraction works with usage object."""
    mock_response = MagicMock()
    mock_response.output_parsed = SimpleResponse(answer="Paris", confidence=0.95)
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 25
    mock_response.usage.output_tokens = 15
    mock_response.usage.total_tokens = 40

    covenance.clear_records()

    with patch("covenance.clients.openai_client.client.responses.parse") as mock_openai:
        mock_openai.return_value = mock_response

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            model="gpt-4o",
        )

    assert isinstance(result, SimpleResponse)
    records = covenance.get_records()
    assert records[-1].tokens_total == 40


def test_openai_usage_extraction_raises_when_missing():
    """Form test: verifies OpenAI usage extraction raises exception when usage information is missing."""

    # Create a mock response without usage attributes using spec
    class ResponseSpec:
        output_parsed = None

    mock_response = MagicMock(spec=ResponseSpec)
    mock_response.output_parsed = SimpleResponse(answer="Paris", confidence=0.95)
    # usage attribute won't exist due to spec, so hasattr will return False

    with patch("covenance.clients.openai_client.client.responses.parse") as mock_openai:
        mock_openai.return_value = mock_response

        with pytest.raises(AttributeError, match="OpenAI response missing usage info"):
            covenance.ask_llm(
                user_msg="What is the capital of France?",
                response_type=SimpleResponse,
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

    def mock_ask_gemini(*args, **kwargs):
        nonlocal candidate_index
        # Check if this is an integration call by looking at user_msg
        user_msg = kwargs.get("user_msg", args[0] if args else "")
        # Integration call contains "candidate answers" text
        if (
            "candidate answers" in user_msg.lower()
            and "worker llms" in user_msg.lower()
        ):
            call_count["integration"] += 1
            result = integration_response
        else:
            call_count["candidate"] += 1
            result = candidate_responses[candidate_index % len(candidate_responses)]
            candidate_index += 1
        # Provider functions always return RawCallResult
        return RawCallResult(output=result, usage=_mock_usage())

    with patch("covenance.clients.google_client.ask_gemini") as mock_ask:
        mock_ask.side_effect = mock_ask_gemini

        result = covenance.llm_consensus(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
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

    def mock_ask_gemini(*args, **kwargs):
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
            result = integration_response
        else:
            result = candidate_response
        # Provider functions always return RawCallResult
        return RawCallResult(output=result, usage=_mock_usage())

    with patch("covenance.clients.google_client.ask_gemini") as mock_ask:
        mock_ask.side_effect = mock_ask_gemini

        covenance.llm_consensus(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
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

    def mock_ask_openai(*args, **kwargs):
        # Capture worker call arguments
        user_msg = kwargs.get("user_msg", args[0] if args else "")
        # Integration call contains "candidate answers" text
        if (
            "candidate answers" in user_msg.lower()
            and "worker llms" in user_msg.lower()
        ):
            result = integration_response
        else:
            # Capture worker call model
            worker_calls.append(
                {
                    "model": kwargs.get("model", args[3] if len(args) > 3 else None),
                }
            )
            result = candidate_response
        # Provider functions always return RawCallResult
        return RawCallResult(output=result, usage=_mock_usage())

    # Models without special prefixes route to OpenAI
    with patch("covenance.clients.openai_client.ask_openai") as mock_ask:
        mock_ask.side_effect = mock_ask_openai

        covenance.llm_consensus(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            sys_msg="Answer concisely.",
            model="gpt-4o",  # Base model (not used when additional_models provided)
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

    def mock_ask_gemini(*args, **kwargs):
        # Capture worker call arguments
        user_msg = kwargs.get("user_msg", args[0] if args else "")
        # Integration call contains "candidate answers" text
        if (
            "candidate answers" in user_msg.lower()
            and "worker llms" in user_msg.lower()
        ):
            result = integration_response
        else:
            # Capture worker call model
            worker_calls.append(
                {
                    "model": kwargs.get("model", args[3] if len(args) > 3 else None),
                }
            )
            result = candidate_response
        # Provider functions always return RawCallResult
        return RawCallResult(output=result, usage=_mock_usage())

    with patch("covenance.clients.google_client.ask_gemini") as mock_ask:
        mock_ask.side_effect = mock_ask_gemini

        covenance.llm_consensus(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
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

    def mock_ask_gemini(*args, **kwargs):
        # Capture worker call arguments
        user_msg = kwargs.get("user_msg", args[0] if args else "")
        # Integration call contains "candidate answers" text
        if (
            "candidate answers" in user_msg.lower()
            and "worker llms" in user_msg.lower()
        ):
            result = integration_response
        else:
            # Capture worker call model
            worker_calls.append(
                {
                    "model": kwargs.get("model", args[3] if len(args) > 3 else None),
                }
            )
            result = candidate_response
        # Provider functions always return RawCallResult
        return RawCallResult(output=result, usage=_mock_usage())

    with patch("covenance.clients.google_client.ask_gemini") as mock_ask:
        mock_ask.side_effect = mock_ask_gemini

        covenance.llm_consensus(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
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
    mock_response_success.usage_metadata.cached_content_token_count = 0

    mock_response_failure = MagicMock()
    mock_response_failure.parsed = None  # Simulate parsing error
    mock_response_failure.usage_metadata = MagicMock()
    mock_response_failure.usage_metadata.prompt_token_count = 10
    mock_response_failure.usage_metadata.candidates_token_count = 5
    mock_response_failure.usage_metadata.total_token_count = 15
    mock_response_failure.usage_metadata.cached_content_token_count = 0

    with patch(
        "covenance.clients.google_client.client.models.generate_content"
    ) as mock_gemini:
        # First call fails, second succeeds
        mock_gemini.side_effect = [mock_response_failure, mock_response_success]

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
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
    mock_response_failure.usage_metadata.cached_content_token_count = 0

    with patch(
        "covenance.clients.google_client.client.models.generate_content"
    ) as mock_gemini:
        # All calls fail
        mock_gemini.return_value = mock_response_failure

        with pytest.raises(StructuredOutputParsingError):
            covenance.ask_llm(
                user_msg="What is the capital of France?",
                response_type=SimpleResponse,
                model="gemini-2.5-flash",
                max_parsing_retries=2,
            )

    # Should have tried 3 times (1 initial + 2 retries)
    assert mock_gemini.call_count == 3


def test_retry_tracking_with_multiple_failures():
    """Comprehensive test: verifies retry tracking when LLM returns invalid JSON twice then succeeds.

    Scenario:
        - Attempt 1: LLM returns None (parsed=None), tokens: 100 in, 50 out
        - Attempt 2: LLM returns None (parsed=None), tokens: 100 in, 60 out
        - Attempt 3: LLM returns valid response, tokens: 100 in, 40 out → success

    Expected record:
        - structured_output_retries = 2 (two failed attempts before success)
        - tokens_input = 300 (all attempts: 100 + 100 + 100)
        - tokens_output = 150 (all attempts: 50 + 60 + 40)
        - Single consolidated record (not 3 separate records)
    """

    def make_mock_response(parsed_value, prompt_tokens: int, completion_tokens: int):
        """Helper to create a mock Gemini API response."""
        response = MagicMock()
        response.parsed = parsed_value
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = prompt_tokens
        response.usage_metadata.candidates_token_count = completion_tokens
        response.usage_metadata.total_token_count = prompt_tokens + completion_tokens
        response.usage_metadata.cached_content_token_count = 0
        return response

    # First two calls fail (parsed=None), third succeeds
    mock_responses = [
        make_mock_response(None, prompt_tokens=100, completion_tokens=50),
        make_mock_response(None, prompt_tokens=100, completion_tokens=60),
        make_mock_response(
            SimpleResponse(answer="Paris", confidence=0.95),
            prompt_tokens=100,
            completion_tokens=40,
        ),
    ]

    covenance.clear_records()

    with patch(
        "covenance.clients.google_client.client.models.generate_content"
    ) as mock_gemini:
        mock_gemini.side_effect = mock_responses

        result = covenance.ask_llm(
            user_msg="What is the capital of France?",
            response_type=SimpleResponse,
            model="gemini-2.5-flash",
            max_parsing_retries=2,  # Allow 2 retries (3 total attempts)
        )

    # Verify successful result
    assert isinstance(result, SimpleResponse)
    assert result.answer == "Paris"
    assert result.confidence == 0.95
    assert mock_gemini.call_count == 3

    # Verify single consolidated record (not 3 separate records)
    records = covenance.get_records()
    assert len(records) == 1, "Expected single consolidated record, not one per attempt"

    record = records[0]

    # Verify retry tracking
    assert record.structured_output_retries == 2, (
        "Expected 2 SO retries (2 failures before success)"
    )
    assert record.tpm_retries == 0, "No TPM retries in this test"

    # Verify total tokens (all attempts combined)
    assert record.tokens_input == 300, (
        "Total input: 100 + 100 + 100 from all 3 attempts"
    )
    assert record.tokens_output == 150, "Total output: 50 + 60 + 40 from all 3 attempts"
    assert record.tokens_total == 450, "Total: 300 in + 150 out"

    # Verify the summary also reflects the retry stats
    summary = covenance.usage_summary()
    assert summary["structured_output_retries"] == 2


