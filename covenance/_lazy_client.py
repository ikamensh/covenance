"""Lazy client proxies to avoid SDK init at import time."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any


class LazyClient:
    """Proxy that creates the real client only when a method is called."""

    def __init__(self, factory: Callable[[], Any], label: str | None = None) -> None:
        self._factory = factory
        self._label = label or getattr(factory, "__name__", "client")
        self._client: Any | None = None
        self._lock = Lock()
        self._children: dict[str, _LazyAttrProxy] = {}

    def resolve(self) -> Any:
        """Instantiate and return the underlying client."""
        return self._get_client()

    def _get_client(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._factory()
        return self._client

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        child = self._children.get(name)
        if child is None:
            child = _LazyAttrProxy(self, (name,))
            self._children[name] = child
        return child

    def __repr__(self) -> str:
        state = "ready" if self._client is not None else "lazy"
        return f"<LazyClient {self._label} ({state})>"


class _LazyAttrProxy:
    def __init__(self, root: LazyClient, path: tuple[str, ...]) -> None:
        self._root = root
        self._path = path
        self._children: dict[str, _LazyAttrProxy] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        child = self._children.get(name)
        if child is None:
            child = _LazyAttrProxy(self._root, self._path + (name,))
            self._children[name] = child
        return child

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        target: Any = self._root._get_client()
        for name in self._path:
            target = getattr(target, name)
        if not callable(target):
            raise TypeError(f"{'.'.join(self._path)} is not callable")
        return target(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<LazyAttrProxy {'.'.join(self._path)}>"
