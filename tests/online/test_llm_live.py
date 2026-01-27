"""Online integration tests for LLM providers."""

import pytest
from pydantic import BaseModel

import covenance

pytestmark = pytest.mark.online


class MathResponse(BaseModel):
    """Response schema for arithmetic checks."""

    result: int


@pytest.fixture(autouse=True)
def reset_records():
    """Reset records before and after each test."""
    covenance.clear_records()
    yield
    covenance.clear_records()


def _assert_usage(provider: str, min_calls: int) -> None:
    records = covenance.get_records()
    assert len(records) >= min_calls
    providers = {record.provider for record in records}
    assert providers == {provider}


def test_online_openai_structured_math(unblock_llm):
    """Integration test: OpenAI structured output parses into schema."""
    result = covenance.ask_llm(
        user_msg="Compute 19 + 23. Return the integer result.",
        format=MathResponse,
        sys_msg="You are a precise calculator. Follow the response schema exactly.",
        model="gpt-5-nano",
    )

    assert isinstance(result, MathResponse)
    assert result.result == 42
    _assert_usage(provider="openai", min_calls=1)


def test_online_gemini_structured_math(unblock_llm):
    """Integration test: Gemini structured output parses into schema."""
    result = covenance.ask_llm(
        user_msg="Compute 19 + 23. Return the integer result.",
        format=MathResponse,
        sys_msg="You are a precise calculator. Follow the response schema exactly.",
        model="gemini-2.5-flash-lite",
    )

    assert isinstance(result, MathResponse)
    assert result.result == 42
    _assert_usage(provider="gemini", min_calls=1)


def test_online_openai_consensus_math(unblock_llm):
    """Integration test: consensus call returns structured output."""
    result = covenance.llm_consensus(
        user_msg="Compute 19 + 23. Return the integer result.",
        format=MathResponse,
        sys_msg="You are a precise calculator. Follow the response schema exactly.",
        model="gpt-5-nano",
        num_candidates=2,
        parallel=False,
    )

    assert isinstance(result, MathResponse)
    assert result.result == 42
    _assert_usage(provider="openai", min_calls=3)
