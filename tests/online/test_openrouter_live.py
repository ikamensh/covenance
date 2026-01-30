"""Simple online integration test for OpenRouter."""

import pytest
from pydantic import BaseModel

import covenance
from covenance.models import OpenRouterModels

pytestmark = pytest.mark.online


class MathResponse(BaseModel):
    result: int


@pytest.fixture(autouse=True)
def reset_records():
    covenance.clear_records()
    yield
    covenance.clear_records()


def test_openrouter_structured_math(unblock_llm):
    """OpenRouter structured output parses into schema."""
    result = covenance.ask_llm(
        "Compute 19 + 23",
        OpenRouterModels.llama_31_8b,
        MathResponse,
    )
    assert isinstance(result, MathResponse)
    assert result.result == 42

    records = covenance.get_records()
    assert len(records) >= 1
    assert all(r.provider == "openrouter" for r in records)
