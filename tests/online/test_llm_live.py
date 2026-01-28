"""Online integration tests for LLM providers."""

import pytest
from pydantic import BaseModel

import covenance
from covenance.models import ClaudeModels, GrokModels

pytestmark = pytest.mark.online


class MathResponse(BaseModel):
    result: int


PROVIDERS = [
    ("gpt-5-nano", "openai"),
    ("gemini-2.5-flash-lite", "gemini"),
    ("mistral-small-latest", "mistral"),
    (ClaudeModels.haiku45, "anthropic"),
    (GrokModels.grok3_mini, "grok"),  # non-reasoning for faster/cheaper tests
]


@pytest.fixture(autouse=True)
def reset_records():
    covenance.clear_records()
    yield
    covenance.clear_records()


@pytest.mark.parametrize("model,provider", PROVIDERS)
def test_structured_math(unblock_llm, model, provider):
    """Structured output parses into schema."""
    result = covenance.ask_llm("Compute 19 + 23", model, MathResponse)
    assert isinstance(result, MathResponse)
    assert result.result == 42

    records = covenance.get_records()
    assert len(records) >= 1
    assert all(r.provider == provider for r in records)


def test_consensus_math(unblock_llm):
    """Consensus call integrates multiple candidates."""
    result = covenance.llm_consensus(
        "Compute 19 + 23",
        "gemini-2.5-flash-lite",
        MathResponse,
        num_candidates=2,
        parallel=False,
    )
    assert isinstance(result, MathResponse)
    assert result.result == 42

    # 2 candidates + 1 integration = 3 calls
    records = covenance.get_records()
    assert len(records) >= 3
