# covenance

Unified, structured LLM calls for OpenAI, Gemini, Mistral, Anthropic, and OpenRouter.

## API keys
Set environment variables:
- OPENAI_API_KEY
- GOOGLE_API_KEY (or GEMINI_API_KEY)
- MISTRAL_API_KEY
- ANTHROPIC_API_KEY
- OPENROUTER_API_KEY
If a `.env` file is present in the working directory, it is loaded automatically
without overriding existing environment variables.

## Call logging
- LLM call timing records are always captured; access in-process via `covenance.get_records()`.
- Persist records by setting `COVENANCE_RECORDS_DIR` or calling `covenance.set_llm_call_records_dir(...)`
  (records are appended to `llm_call_records.jsonl` in that folder).
- To visualize, run `python scripts/export_llm_calls.py` then open `scripts/llm_calls.html` in a browser.

## Clients (separate keys + history)
Use `Covenance` to isolate API keys and call records per task or subsystem.

```
from covenance import Covenance

client = Covenance(
    label="risk-review",
    openai_api_key="sk-...",
    records_dir="/tmp/my_records",  # optional: persist to JSONL
)
result = client.ask_llm("Summarize", model="gpt-5")
records = client.get_records()
```

Module-level helpers (`covenance.ask_llm`, `covenance.llm_consensus`, `covenance.get_records`)
use the default instance.
