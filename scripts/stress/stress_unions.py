"""Stress test: union and discriminated union types.

Tests type-heterogeneous fields — a key differentiator between backends
since schema translation for unions varies significantly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from stress_utils import DEFAULT_MODEL, StressTestResult, make_client, run_test_cases


# Simple union: field can be str or int
class FlexibleRecord(BaseModel):
    label: str
    value: int | str = Field(
        description="Either an integer count or a string description"
    )


# Optional with union
class SearchResult(BaseModel):
    title: str
    score: float | None = None
    snippet: str | None = None


# Discriminated union via tagged models
class CatProfile(BaseModel):
    animal_type: Literal["cat"]
    name: str
    indoor: bool


class DogProfile(BaseModel):
    animal_type: Literal["dog"]
    name: str
    breed: str


class FishProfile(BaseModel):
    animal_type: Literal["fish"]
    name: str
    freshwater: bool


class PetRecord(BaseModel):
    owner: str
    pet: CatProfile | DogProfile | FishProfile


# List of union items
class MixedList(BaseModel):
    items: list[CatProfile | DogProfile]


# Nested union
class Response(BaseModel):
    class TextContent(BaseModel):
        kind: Literal["text"]
        body: str

    class NumberContent(BaseModel):
        kind: Literal["number"]
        value: float

    content: TextContent | NumberContent
    metadata: str


def validate_flexible(result: FlexibleRecord) -> tuple[bool, str]:
    if not result.label:
        return False, "label empty"
    if not isinstance(result.value, (int, str)):
        return False, f"value type {type(result.value)}, expected int|str"
    return True, ""


def validate_search(result: SearchResult, expect_filled: bool) -> tuple[bool, str]:
    if not result.title:
        return False, "title empty"
    if expect_filled and result.score is None:
        return False, "score should be filled"
    return True, ""


def validate_pet_cat(result: PetRecord) -> tuple[bool, str]:
    if not result.owner:
        return False, "owner empty"
    if not isinstance(result.pet, CatProfile):
        return False, f"Expected CatProfile, got {type(result.pet).__name__}"
    if result.pet.animal_type != "cat":
        return False, f"Expected animal_type='cat', got '{result.pet.animal_type}'"
    return True, ""


def validate_pet_dog(result: PetRecord) -> tuple[bool, str]:
    if not result.owner:
        return False, "owner empty"
    if not isinstance(result.pet, DogProfile):
        return False, f"Expected DogProfile, got {type(result.pet).__name__}"
    if result.pet.animal_type != "dog":
        return False, f"Expected animal_type='dog', got '{result.pet.animal_type}'"
    if not result.pet.breed:
        return False, "breed empty"
    return True, ""


def validate_mixed_list(result: MixedList) -> tuple[bool, str]:
    if len(result.items) < 3:
        return False, f"Expected >=3 items, got {len(result.items)}"
    types_seen = {type(item).__name__ for item in result.items}
    if len(types_seen) < 2:
        return False, f"Expected mix of types, only saw {types_seen}"
    return True, ""


def validate_response_text(result: Response) -> tuple[bool, str]:
    if not isinstance(result.content, Response.TextContent):
        return False, f"Expected TextContent, got {type(result.content).__name__}"
    if result.content.kind != "text":
        return False, f"Expected kind='text', got '{result.content.kind}'"
    if not result.content.body:
        return False, "body empty"
    return True, ""


def run_stress_test(
    model: str = DEFAULT_MODEL, backend: str | None = None
) -> StressTestResult:
    """Test union type handling."""
    client = make_client(model, backend)

    cases = [
        (
            "simple_union",
            lambda: client.ask_llm(
                "Create a record with label 'count' and an integer value of 42.",
                model=model,
                response_type=FlexibleRecord,
            ),
            validate_flexible,
        ),
        (
            "optional_filled",
            lambda: client.ask_llm(
                "Create a search result with title, score (0.95), and a snippet.",
                model=model,
                response_type=SearchResult,
            ),
            lambda r: validate_search(r, expect_filled=True),
        ),
        (
            "discriminated_cat",
            lambda: client.ask_llm(
                "Create a pet record for an owner named Alice who has an indoor cat named Whiskers. "
                "The animal_type MUST be 'cat'.",
                model=model,
                response_type=PetRecord,
            ),
            validate_pet_cat,
        ),
        (
            "discriminated_dog",
            lambda: client.ask_llm(
                "Create a pet record for owner Bob with a golden retriever named Rex. "
                "The animal_type MUST be 'dog'.",
                model=model,
                response_type=PetRecord,
            ),
            validate_pet_dog,
        ),
        (
            "union_list",
            lambda: client.ask_llm(
                "Create a list of 4 pets: 2 cats and 2 dogs. Each must have the correct animal_type.",
                model=model,
                response_type=MixedList,
            ),
            validate_mixed_list,
        ),
        (
            "nested_union",
            lambda: client.ask_llm(
                "Create a response with text content (kind='text') saying 'Hello world' "
                "and metadata 'greeting'.",
                model=model,
                response_type=Response,
            ),
            validate_response_text,
        ),
    ]

    return run_test_cases(client, "unions", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
