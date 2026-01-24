## LLM Structured Calls Package
Intention: release this as a standalone, maximally useful package focusing on:
simple signature to call llms of different providers with structured output
+ utilities like retry, consensus, stats"


### Scope
Provide a simple, provider-agnostic interface for structured LLM calls
(`ask_llm_structured`, `ask_llm_structured_with_consensus`) plus utilities
like retry, consensus, and usage/metrics tracking.

### What was copied
- `online_llm/` from `autodpia/lib/online_llm`

### Current interdependencies
- Provider clients import API keys from environment variables.
- Model enums are defined inside provider modules, which also create SDK clients.
- `LLMModelName` is built by importing provider enums, which triggers client init.
- Metrics and usage tracking are embedded in the core flow
  (`metrics.py`, `usage.py`), with optional context propagation.

### Gaps for a standalone package
- SDK clients are created at import time, which requires keys even when unused.
- Model enums couple type definitions to provider implementations.
- Routing is purely string-based (prefix and "/" heuristics), no explicit provider.
- Request options are narrow (no standardized `temperature`, `max_tokens`, etc.).
- Metrics are bound to Cloud Run conventions (host from `K_SERVICE`).

### Suggested fixes
- Make client creation lazy and keyed on first use.
- Split model definitions from client modules; build `LLMModelName` without imports
  that create clients.
- Add a provider override in `ask_llm_structured` (e.g., `provider="openai"`).
- Introduce a small `RequestOptions` object for standard knobs and pass-through
  `provider_kwargs`.
- Make metrics optional/pluggable; default to no-op if unused.

### Next steps
- Decide target package name and top-level module layout.
- Extract code into a new repo, then update imports in consumers.
- Add minimal tests for routing, retry, and consensus in the new package.




