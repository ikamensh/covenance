# covenance

[![PyPI version](https://img.shields.io/pypi/v/covenance)](https://pypi.org/project/covenance/)
[![Tests](https://github.com/ikamensh/covenance/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/ikamensh/covenance/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ikamensh/covenance/branch/main/graph/badge.svg)](https://codecov.io/gh/ikamensh/covenance)

Type-safe LLM outputs across any provider. Track every call and its cost.

```python
from covenance import ask_llm

review = ask_llm("Write a short review of Inception", model="gpt-4.1-nano")
is_positive = ask_llm(f"Is this review positive? '{review}'", model="gemini-2.5-flash-lite", response_type=bool)
print(is_positive)  # True
```

## Usecases

- **Structured outputs that work** - Same code, any provider. Pydantic models, primitives, lists, tuples.
- **Zero routing config** - Model name determines provider automatically (`gemini-*`, `claude-*`, `gpt-*`)
- **Know what you're spending** - Every call logged with token counts and cost. `print_usage()` for totals, `print_call_timeline()` for a visual waterfall.

## Installation

Install only the providers you need:

```bash
pip install covenance[openai]      # OpenAI, Grok, OpenRouter
pip install covenance[anthropic]   # Anthropic Claude
pip install covenance[google]      # Google Gemini
pip install covenance[mistral]     # Mistral

# Multiple providers
pip install covenance[openai,anthropic]

# All providers
pip install covenance[all]
```

## Structured outputs

Pass `response_type` to get validated, typed results:

```python
# Pydantic models
class Evaluation(BaseModel):
    reasoning: str
    is_correct: bool

result = ask_llm("Is 2+2=5?", model="gemini-2.5-flash-lite", response_type=Evaluation)
print(result.reasoning)  # "2+2 equals 4, not 5"
print(result.is_correct)  # False

# Primitives
answer = ask_llm("Is Python interpreted?", model="gpt-4.1-nano", response_type=bool)
print(answer)  # True

# Collections
items = ask_llm("List 3 prime numbers", model="claude-sonnet-4-20250514", response_type=list[int])
print(items)  # [2, 3, 5]
```

Works identically across OpenAI, Gemini, Anthropic, Mistral, Grok, and OpenRouter.

## Cost tracking

Every call is recorded with token counts and cost:

```python
from covenance import ask_llm, print_usage, print_call_timeline, get_records

ask_llm("Hello", model="gpt-4.1-nano")
ask_llm("Hello", model="gemini-2.5-flash-lite")

print_usage()
# ==================================================
# LLM Usage Summary (default client)
# ==================================================
#   Calls: 2
#   Tokens: 45 (In: 12, Out: 33)
#   Cost: $0.0001
#   Models: gemini/gemini-2.5-flash-lite, openai/gpt-4.1-nano

# Access individual records
for record in get_records():
    print(f"{record.model}: {record.cost_usd}")
```

Persist records by setting `COVENANCE_RECORDS_DIR` or calling `set_llm_call_records_dir()`.

## Call timeline

Visualize call sequences and parallelism in your terminal:

```python
from covenance import print_call_timeline

print_call_timeline()
# LLM Call Timeline (4.4s total, 5 calls)
#                         |0s                                            4.4s|
#   gpt-4.1-nano    1.3s  |████████████████                                  |
#   g2.5-flash-l    1.1s  |                 ████████████                     |
#   g2.5-flash-l    1.1s  |                 ████████████                     |
#   g2.5-flash-l    1.5s  |                 ████████████████                 |
#   g2.5-flash-l    1.5s  |                                 █████████████████|
```

Each line is a call, sorted by start time. Blocks show when each call was active - parallel calls appear as overlapping bars on different rows.

## Consensus for quality

Run parallel LLM calls and integrate results for higher quality:

```python
from covenance import llm_consensus

result = llm_consensus(
    "Explain quantum entanglement",
    model="gpt-4.1-nano",
    response_type=Evaluation,
    num_candidates=3,  # 3 parallel calls + integration
)
```

## Supported providers

Provider is determined by model name prefix:

| Prefix | Provider |
|--------|----------|
| `gpt-*`, `o1-*`, `o3-*` | OpenAI |
| `gemini-*` | Google Gemini |
| `claude-*` | Anthropic |
| `mistral-*`, `codestral-*` | Mistral |
| `grok-*` | xAI Grok |
| `org/model` (contains `/`) | OpenRouter |

### Structured output reliability

Providers differ in how they enforce JSON schema compliance:

| Provider | Method | Guarantee |
|----------|--------|-----------|
| OpenAI | [Constrained decoding](https://openai.com/index/introducing-structured-outputs-in-the-api) | 100% schema-valid JSON |
| Google Gemini | [Controlled generation](https://ai.google.dev/gemini-api/docs/structured-output) | 100% schema-valid JSON |
| Grok | [Constrained decoding](https://docs.x.ai/docs/guides/structured-outputs) | 100% schema-valid JSON |
| Anthropic | [Structured outputs beta](https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs) | 100% schema-valid JSON* |
| Mistral | [Best-effort](https://docs.mistral.ai/capabilities/structured_output) | Probabilistic |
| OpenRouter | Varies | Depends on underlying model |

*Anthropic structured outputs requires SDK >= 0.74.1 (uses `anthropic-beta: structured-outputs-2025-11-13`). Mistral uses probabilistic generation. Covenance retries automatically (up to 3 times) on JSON parse errors for Mistral.

## API keys

Set environment variables for the providers you use:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)
- `ANTHROPIC_API_KEY`
- `MISTRAL_API_KEY`
- `OPENROUTER_API_KEY`
- `XAI_API_KEY` (for Grok)

A `.env` file in the working directory is loaded automatically.

## Isolated clients

Use `Covenance` instances for separate API keys and call records per subsystem:

```python
from covenance import Covenance
from pydantic import BaseModel

# Each client tracks its own usage
question_client = Covenance(label="questions")
review_client = Covenance(label="review")

answer = question_client.ask_llm("Who is David Blaine?", model="gpt-4.1-nano")

class Evaluation(BaseModel):
    reasoning: str
    is_correct: bool

eval = review_client.llm_consensus(
    f"Is this accurate? '''{answer}'''",
    model="gemini-2.5-flash-lite",
    response_type=Evaluation,
)

question_client.print_usage()  # Shows only the question call
review_client.print_usage()    # Shows only the review call
```
