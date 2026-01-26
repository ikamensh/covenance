"""Context-scoped overrides for LLM clients and call records."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .record import RecordStore, get_default_record_store

_record_store: ContextVar[RecordStore | None] = ContextVar(
    "covenance_record_store", default=None
)
_client_overrides: ContextVar[dict[str, Any] | None] = ContextVar(
    "covenance_client_overrides", default=None
)


def get_record_store() -> RecordStore:
    store = _record_store.get()
    return store if store is not None else get_default_record_store()


def get_client_override(provider: str) -> Any | None:
    overrides = _client_overrides.get()
    if overrides is None:
        return None
    return overrides.get(provider)


def snapshot() -> tuple[RecordStore | None, dict[str, Any] | None]:
    return (_record_store.get(), _client_overrides.get())


def restore(state: tuple[RecordStore | None, dict[str, Any] | None]) -> None:
    record_store, client_overrides = state
    _record_store.set(record_store)
    _client_overrides.set(client_overrides)


@contextmanager
def use_instance_context(
    record_store: RecordStore | None,
    client_overrides: dict[str, Any] | None,
):
    token_store = _record_store.set(record_store)
    token_clients = _client_overrides.set(client_overrides)
    try:
        yield
    finally:
        _client_overrides.reset(token_clients)
        _record_store.reset(token_store)

