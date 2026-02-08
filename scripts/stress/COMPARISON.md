# Stress Test Comparison: Main vs Pydantic-AI

## Summary

| Branch | Total Pass | gpt-4.1-nano | gpt-4.1-mini | gemini-2.5-lite | gemini-2.0-flash | mistral-small |
|--------|------------|--------------|--------------|-----------------|------------------|---------------|
| **Main (internal)** | 52/60 | 12/12 | 11/12 | 9/12 | 9/12 | 11/12 |
| **Pydantic-AI**     | 51/60 | 9/12  | 11/12 | 11/12 | 11/12 | 9/12 |

## Side-by-Side Test Results

| Test | Main: nano | PyAI: nano | Main: mini | PyAI: mini | Main: lite | PyAI: lite | Main: flash | PyAI: flash | Main: mistral | PyAI: mistral |
|------|------------|------------|------------|------------|------------|------------|-------------|-------------|---------------|---------------|
| nesting_depth      | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | **FAIL** |
| field_width        | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| recursive          | PASS | **FAIL** | PASS | PASS | **FAIL** | PASS | **FAIL** | PASS | PASS | PASS |
| type_diversity     | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| enums              | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| optionals          | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| field_constraints  | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| cross_field        | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| edge_values        | PASS | PASS | **FAIL** | PASS | PASS | PASS | PASS | PASS | PASS | **FAIL** |
| consistency        | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| real_schemas       | PASS | **FAIL** | PASS | PASS | **FAIL** | PASS | **FAIL** | PASS | **FAIL** | PASS |
| limits             | PASS | **FAIL** | PASS | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | PASS | **FAIL** |

## Key Differences

### Tests that got WORSE with pydantic-ai:
- **limits**: 3/5 → 0/5 (all models now fail)
- **gpt-4.1-nano**: recursive (new fail), real_schemas (new fail), limits (new fail)
- **mistral**: nesting_depth (new fail), edge_values (new fail), limits (new fail)

### Tests that got BETTER with pydantic-ai:
- **recursive**: gemini-2.5-lite and gemini-2.0-flash now pass (were failing)
- **real_schemas**: gemini-2.5-lite, gemini-2.0-flash, mistral now pass (were failing)

### Analysis

**The limits test is the biggest concern** - it universally fails with pydantic-ai. This suggests pydantic-ai has a lower ceiling for schema complexity/size.

**Trade-off pattern:**
- Main branch: Better with OpenAI models, worse with Gemini models
- Pydantic-AI: Better with Gemini models, worse with OpenAI nano and Mistral

**OpenAI gpt-4.1-nano regressed significantly** (12/12 → 9/12), losing recursive, real_schemas, and limits.

**Gemini models improved** (9/12 → 11/12 each), gaining recursive and real_schemas.

## Root Cause Analysis

### Why `limits` fails universally on pydantic-ai

The two implementations use different APIs:

| Branch | API Used | Schema Handling |
|--------|----------|-----------------|
| **Main** | OpenAI `responses.parse()` | OpenAI handles schema server-side |
| **Pydantic-ai** | Chat completions + `response_format` | Raw JSON Schema sent in request body |

Pydantic generates verbose JSON schemas - enum values are repeated inline for every field instead of using `$defs` references:

```
30 fields × 20 enum values:  ~9KB  → PASS
50 fields × 20 enum values: ~15KB  → PASS  
100 fields × 20 enum values: ~31KB → FAIL (all providers)
```

The chat completions endpoint has stricter schema size/complexity limits than the Responses API.

### Why Gemini improved with pydantic-ai

Pydantic-ai has better-maintained Gemini integration. The main branch's hand-rolled client likely has issues with:
- Recursive type handling (both Gemini models failed `recursive` on main)
- Schema generation for complex nested types (`real_schemas` failed on main)

### Why OpenAI nano and Mistral regressed

These smaller/cheaper models are more sensitive to how the schema is presented. The chat completions `response_format` path may produce schemas that are harder for these models to follow correctly compared to the native Responses API.

## Trade-off Summary

- **Main branch**: Direct access to newer APIs (Responses API) with higher schema limits, but more maintenance burden per provider
- **Pydantic-ai**: Better-maintained provider adapters (especially Gemini), but constrained by chat completions schema limits
