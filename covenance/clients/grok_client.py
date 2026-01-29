"""xAI Grok client using OpenAI-compatible API."""

from typing import TYPE_CHECKING

from covenance._lazy_client import LazyClient
from covenance.clients.openai_client import ask_openai_compatible_structured
from covenance.exceptions import require_provider
from covenance.keys import get_grok_api_key, require_api_key
from covenance.models import GrokModels

if TYPE_CHECKING:
    from openai import OpenAI

    from covenance.record import RecordStore

GROK_BASE_URL = "https://api.x.ai/v1"


def _create_grok_client() -> "OpenAI":
    require_provider("openai")
    from openai import OpenAI

    api_key = require_api_key(get_grok_api_key(), "grok")
    return OpenAI(api_key=api_key, base_url=GROK_BASE_URL)


client = LazyClient(_create_grok_client, label="grok")


def ask_grok[T](
    user_msg: str,
    response_type: type[T] | None = None,
    sys_msg: str | None = None,
    model: str = GrokModels.grok4_fast.value,
    *,
    client_override: "OpenAI | None" = None,
    record_store: "RecordStore | None" = None,
    temperature: float | None = None,
) -> T:
    """Call xAI Grok API with automatic retry."""
    api_client = client_override or client  # type: ignore[assignment]
    return ask_openai_compatible_structured(
        client=api_client,
        user_msg=user_msg,
        response_type=response_type,
        sys_msg=sys_msg,
        model=model,
        provider="grok",
        record_store=record_store,
        temperature=temperature,
    )
