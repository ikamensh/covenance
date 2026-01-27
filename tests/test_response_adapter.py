"""Unit tests for ResponseTypeAdapter."""

from pydantic import BaseModel

from covenance.response_adapter import ResponseTypeAdapter


class SampleModel(BaseModel):
    name: str
    value: int


class TestResponseTypeAdapterNoWrapping:
    """Tests for types that don't need wrapping."""

    def test_none_type(self):
        adapter = ResponseTypeAdapter(None)
        assert not adapter.needs_wrapping
        assert adapter.get_llm_type() is None
        assert adapter.unwrap("hello") == "hello"

    def test_str_type(self):
        adapter = ResponseTypeAdapter(str)
        assert not adapter.needs_wrapping
        assert adapter.get_llm_type() is str
        assert adapter.unwrap("hello") == "hello"

    def test_pydantic_model(self):
        adapter = ResponseTypeAdapter(SampleModel)
        assert not adapter.needs_wrapping
        assert adapter.get_llm_type() is SampleModel

        model = SampleModel(name="test", value=42)
        assert adapter.unwrap(model) is model


class TestResponseTypeAdapterListWrapping:
    """Tests for list types that need wrapping."""

    def test_list_int(self):
        adapter = ResponseTypeAdapter(list[int])
        assert adapter.needs_wrapping

        wrapper_type = adapter.get_llm_type()
        assert wrapper_type is not None
        assert hasattr(wrapper_type, "model_fields")
        assert "result" in wrapper_type.model_fields

        # Simulate LLM response
        wrapped = wrapper_type(result=[1, 2, 3])
        unwrapped = adapter.unwrap(wrapped)
        assert unwrapped == [1, 2, 3]

    def test_list_str(self):
        adapter = ResponseTypeAdapter(list[str])
        assert adapter.needs_wrapping

        wrapper_type = adapter.get_llm_type()
        wrapped = wrapper_type(result=["a", "b"])
        assert adapter.unwrap(wrapped) == ["a", "b"]

    def test_list_model(self):
        adapter = ResponseTypeAdapter(list[SampleModel])
        assert adapter.needs_wrapping

        wrapper_type = adapter.get_llm_type()
        items = [SampleModel(name="a", value=1), SampleModel(name="b", value=2)]
        wrapped = wrapper_type(result=items)
        assert adapter.unwrap(wrapped) == items


class TestResponseTypeAdapterTupleWrapping:
    """Tests for tuple types that need special wrapping."""

    def test_tuple_two_elements(self):
        adapter = ResponseTypeAdapter(tuple[str, int])
        assert adapter.needs_wrapping

        wrapper_type = adapter.get_llm_type()
        assert "item_0" in wrapper_type.model_fields
        assert "item_1" in wrapper_type.model_fields

        wrapped = wrapper_type(item_0="hello", item_1=42)
        unwrapped = adapter.unwrap(wrapped)
        assert unwrapped == ("hello", 42)
        assert isinstance(unwrapped, tuple)

    def test_tuple_three_elements(self):
        adapter = ResponseTypeAdapter(tuple[str, int, float])
        assert adapter.needs_wrapping

        wrapper_type = adapter.get_llm_type()
        assert "item_0" in wrapper_type.model_fields
        assert "item_1" in wrapper_type.model_fields
        assert "item_2" in wrapper_type.model_fields

        wrapped = wrapper_type(item_0="test", item_1=100, item_2=3.14)
        unwrapped = adapter.unwrap(wrapped)
        assert unwrapped == ("test", 100, 3.14)
        assert isinstance(unwrapped, tuple)

    def test_tuple_with_model(self):
        adapter = ResponseTypeAdapter(tuple[str, SampleModel])
        assert adapter.needs_wrapping

        wrapper_type = adapter.get_llm_type()
        model = SampleModel(name="nested", value=99)
        wrapped = wrapper_type(item_0="prefix", item_1=model)
        unwrapped = adapter.unwrap(wrapped)
        assert unwrapped == ("prefix", model)


class TestResponseTypeAdapterSchemaGeneration:
    """Tests for JSON schema generation."""

    def test_list_schema_has_no_additional_properties(self):
        """list[int] wrapper should not use additionalProperties."""
        adapter = ResponseTypeAdapter(list[int])
        wrapper_type = adapter.get_llm_type()
        schema = wrapper_type.model_json_schema()

        # Should have 'result' property, not additionalProperties
        assert "properties" in schema
        assert "result" in schema["properties"]
        assert "additionalProperties" not in schema

    def test_tuple_schema_has_no_prefix_items(self):
        """tuple[str, int] wrapper should not use prefixItems."""
        adapter = ResponseTypeAdapter(tuple[str, int])
        wrapper_type = adapter.get_llm_type()
        schema = wrapper_type.model_json_schema()

        # Should have item_0, item_1 properties, not prefixItems
        assert "properties" in schema
        assert "item_0" in schema["properties"]
        assert "item_1" in schema["properties"]
        assert "prefixItems" not in schema
