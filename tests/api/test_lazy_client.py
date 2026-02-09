"""Tests for lazy client proxy behavior.

Doctest summary:
>>> calls = {"n": 0}
>>> def factory():
...     calls["n"] += 1
...     return object()
>>> _ = factory()
>>> calls["n"]
1
"""

import pytest

from covenance._lazy_client import LazyClient


class _API:
    def ping(self, value: int) -> int:
        return value + 1


class _Root:
    def __init__(self):
        self.api = _API()
        self.value = 123


def test_resolve_is_lazy_and_cached():
    created = {"count": 0}

    def factory():
        created["count"] += 1
        return _Root()

    client = LazyClient(factory, label="demo")
    assert "lazy" in repr(client)
    assert created["count"] == 0

    first = client.resolve()
    second = client.resolve()
    assert first is second
    assert created["count"] == 1
    assert "ready" in repr(client)


def test_nested_proxy_call_and_child_cache():
    created = {"count": 0}

    def factory():
        created["count"] += 1
        return _Root()

    client = LazyClient(factory)

    assert client.api is client.api
    assert client.api.ping(2) == 3
    assert created["count"] == 1


def test_private_attributes_are_not_exposed():
    client = LazyClient(lambda: _Root())

    with pytest.raises(AttributeError):
        _ = client._internal  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        _ = client.api._internal  # type: ignore[attr-defined]


def test_calling_non_callable_proxy_raises_type_error():
    client = LazyClient(lambda: _Root())

    with pytest.raises(TypeError, match="value is not callable"):
        client.value()


def test_proxy_repr_includes_path():
    client = LazyClient(lambda: _Root())
    assert "api.ping" in repr(client.api.ping)
