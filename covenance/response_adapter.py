"""Response type adapter for LLM structured output.

Handles wrapping arbitrary Python types into Pydantic models for LLM APIs,
and unwrapping the results back to the original types.

Supported types:
- Pydantic models (no wrapping needed)
- list[X] (wrapped in model with 'result' field)
- tuple[X, Y, Z] (wrapped in model with 'item_0', 'item_1', ... fields)
- Other generic types (wrapped in model with 'result' field)
"""

from typing import Any, get_args, get_origin

from pydantic import BaseModel, create_model


def _is_pydantic_model(t: Any) -> bool:
    """Check if type is a Pydantic model class."""
    return isinstance(t, type) and issubclass(t, BaseModel)


def _is_tuple_type(t: Any) -> bool:
    """Check if type is a tuple type hint like tuple[str, int, float]."""
    return get_origin(t) is tuple


class ResponseTypeAdapter:
    """Adapts arbitrary Python types to Pydantic models for LLM structured output.

    Usage:
        adapter = ResponseTypeAdapter(list[int])
        wrapper_type = adapter.get_llm_type()  # Pydantic model to send to LLM
        result = adapter.unwrap(llm_result)    # Extract original type from response
    """

    def __init__(self, response_type: type | None):
        self.original_type = response_type
        self._wrapper_model: type[BaseModel] | None = None
        self._is_tuple = False
        self._tuple_length = 0

        # Determine wrapping strategy
        if response_type is None or response_type is str:
            self._needs_wrapping = False
        elif _is_pydantic_model(response_type):
            self._needs_wrapping = False
        elif _is_tuple_type(response_type):
            self._needs_wrapping = True
            self._is_tuple = True
            self._build_tuple_wrapper(response_type)
        else:
            self._needs_wrapping = True
            self._build_simple_wrapper(response_type)

    def _build_simple_wrapper(self, type_hint: type) -> None:
        """Create wrapper with single 'result' field."""
        self._wrapper_model = create_model("_TypeWrapper", result=(type_hint, ...))

    def _build_tuple_wrapper(self, type_hint: type) -> None:
        """Create wrapper with item_0, item_1, ... fields for tuple elements."""
        args = get_args(type_hint)
        self._tuple_length = len(args)

        fields = {f"item_{i}": (arg, ...) for i, arg in enumerate(args)}
        self._wrapper_model = create_model("_TupleWrapper", **fields)

    def get_llm_type(self) -> type | None:
        """Get the type to pass to the LLM API."""
        if self._needs_wrapping:
            return self._wrapper_model
        return self.original_type

    def unwrap(self, result: Any) -> Any:
        """Unwrap LLM result back to original type."""
        if not self._needs_wrapping:
            return result

        if self._is_tuple:
            # Reconstruct tuple from item_0, item_1, ...
            items = [getattr(result, f"item_{i}") for i in range(self._tuple_length)]
            return tuple(items)
        else:
            return result.result

    @property
    def needs_wrapping(self) -> bool:
        return self._needs_wrapping
