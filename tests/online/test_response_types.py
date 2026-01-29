"""Online tests for response_type support across providers.

Type Support Matrix:
  All:     ✓ Pydantic models, int/float/bool, list[X], list[list[X]], tuple[...]
  Mistral: ✓ dict[str, X] (supports additionalProperties)
  OpenAI/Gemini: ✗ dict[str, X] (rejects additionalProperties)

Run with: pytest -m online tests/online/test_response_types.py -v
"""

import pytest
from pydantic import BaseModel

import covenance

pytestmark = pytest.mark.online


# Test models
class SimpleItem(BaseModel):
    name: str
    value: int


class NestedItem(BaseModel):
    label: str
    numbers: list[int]


# Provider models
# Use mini 2/3 of the time and nano 1/3 to distribute load across TPM limits
MODELS = [
    "gpt-4.1-nano",
    "gemini-2.5-flash-lite",
    "mistral-small-latest",
]
MODELS_NO_DICT = [
    "gpt-4.1-nano",
    "gemini-2.5-flash-lite",
]  # OpenAI/Gemini reject dict types


@pytest.fixture(autouse=True)
def reset_records():
    covenance.clear_records()
    yield
    covenance.clear_records()


# Pydantic models
@pytest.mark.parametrize("model", MODELS)
def test_pydantic_simple(unblock_llm, model):
    result = covenance.ask_llm("Item named 'apple' with value 42", model, SimpleItem)
    assert isinstance(result, SimpleItem)


@pytest.mark.parametrize("model", MODELS)
def test_pydantic_nested(unblock_llm, model):
    result = covenance.ask_llm(
        "Item labeled 'primes' with numbers [2, 3, 5]", model, NestedItem
    )
    assert isinstance(result, NestedItem)
    assert all(isinstance(x, int) for x in result.numbers)


# Native types (auto-wrapped)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "response_type,expected_type",
    [
        (int, int),
        (float, (int, float)),
        (bool, bool),
    ],
)
def test_native_types(unblock_llm, model, response_type, expected_type):
    result = covenance.ask_llm("What is 19 + 23?", model, response_type)
    assert isinstance(result, expected_type)


# List types
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "response_type,elem_type",
    [
        (list[int], int),
        (list[str], str),
        (list[float], (int, float)),
    ],
)
def test_list_types(unblock_llm, model, response_type, elem_type):
    result = covenance.ask_llm("Give me a few examples", model, response_type)
    assert isinstance(result, list)
    assert all(isinstance(x, elem_type) for x in result)


@pytest.mark.parametrize("model", MODELS)
def test_list_pydantic(unblock_llm, model):
    result = covenance.ask_llm(
        "Give me items with names and values", model, list[SimpleItem]
    )
    assert isinstance(result, list) and len(result) >= 1
    assert all(isinstance(x, SimpleItem) for x in result)


@pytest.mark.parametrize("model", MODELS)
def test_list_nested(unblock_llm, model):
    result = covenance.ask_llm("Give me groups of numbers", model, list[list[int]])
    assert isinstance(result, list)


# Tuple types
@pytest.mark.parametrize("model", MODELS)
def test_tuple_mixed(unblock_llm, model):
    result = covenance.ask_llm(
        "A word, an integer, and a decimal", model, tuple[str, int, float]
    )
    assert isinstance(result, tuple) and len(result) == 3
    assert isinstance(result[0], str)
    assert isinstance(result[1], int)


# Dict types - Mistral supports additionalProperties but output is probabilistic (may fail JSON parsing)
# OpenAI/Gemini reject additionalProperties in schema
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            m,
            marks=pytest.mark.xfail(
                reason="additionalProperties rejected", raises=Exception
            ),
        )
        for m in MODELS_NO_DICT
    ]
    + [
        pytest.param(
            "mistral-small-latest",
            marks=pytest.mark.flaky(
                reruns=2
            ),  # Mistral structured output is probabilistic
        )
    ],
)
def test_dict_str_int(unblock_llm, model):
    result = covenance.ask_llm("Mapping: apple=1, banana=2", model, dict[str, int])
    assert isinstance(result, dict)
