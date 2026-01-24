"""Online integration tests for LLM providers."""

import pytest
from pydantic import BaseModel

import covenance
from covenance.usage import usage_stats

pytestmark = pytest.mark.online


class MathResponse(BaseModel):
    """Response schema for arithmetic checks."""

    result: int


@pytest.fixture(autouse=True)
def reset_usage_stats():
    """Reset usage stats before and after each test."""
    usage_stats.reset()
    yield
    usage_stats.reset()


def _assert_usage(provider: str, min_calls: int) -> None:
    records = usage_stats.get_detailed_records()
    assert len(records) >= min_calls
    providers = {record["provider"] for record in records}
    assert providers == {provider}


def test_online_openai_structured_math(unblock_llm):
    """Integration test: OpenAI structured output parses into schema."""
    result = covenance.ask_llm(
        user_msg="Compute 19 + 23. Return the integer result.",
        format=MathResponse,
        sys_msg="You are a precise calculator. Follow the response schema exactly.",
        model="gpt-5-mini",
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
        model="gemini-2.5-flash",
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
        model="gpt-5-mini",
        num_candidates=2,
        parallel=False,
    )

    assert isinstance(result, MathResponse)
    assert result.result == 42
    _assert_usage(provider="openai", min_calls=3)
