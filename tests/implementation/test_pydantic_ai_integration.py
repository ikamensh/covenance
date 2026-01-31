"""Tests for pydantic-ai internal access patterns.

These tests verify that our access to pydantic-ai internal structures
still works. If pydantic-ai changes their internals, these tests will fail
and alert us to update our code.
"""


class TestStructuredOutputRetriesExtraction:
    """Verify we can extract structured output retries from pydantic-ai."""

    def test_agent_run_result_has_state_attribute(self):
        """AgentRunResult must have _state attribute for retry tracking.

        We rely on result._state.retries to track structured output retries.
        This test will fail if pydantic-ai removes or renames this attribute.
        """
        from pydantic_ai.run import AgentRunResult

        # Check that _state field exists in the dataclass
        field_names = [f.name for f in AgentRunResult.__dataclass_fields__.values()]
        assert "_state" in field_names, (
            "pydantic-ai AgentRunResult no longer has _state field. "
            "Our structured_output_retries tracking will not work. "
            "Update client.py to use the new API."
        )

    def test_graph_agent_state_has_retries_attribute(self):
        """GraphAgentState must have retries attribute.

        The _state object must have a retries field that counts
        structured output parsing retries.
        """
        from pydantic_ai._agent_graph import GraphAgentState

        # Check that retries field exists in the dataclass
        field_names = [f.name for f in GraphAgentState.__dataclass_fields__.values()]
        assert "retries" in field_names, (
            "pydantic-ai GraphAgentState no longer has retries field. "
            "Our structured_output_retries tracking will not work. "
            "Update client.py to use the new API."
        )

    def test_retries_field_is_int_defaulting_to_zero(self):
        """GraphAgentState.retries should be an int defaulting to 0."""
        from pydantic_ai._agent_graph import GraphAgentState

        state = GraphAgentState()
        assert isinstance(state.retries, int)
        assert state.retries == 0

    def test_extraction_pattern_works(self):
        """Our getattr extraction pattern should work on real objects.

        This tests the exact pattern we use in client.py:
            structured_retries = getattr(result, "_state", None)
            structured_retries = getattr(structured_retries, "retries", 0) if structured_retries else 0
        """
        from pydantic_ai._agent_graph import GraphAgentState
        from pydantic_ai.run import AgentRunResult

        # Create a minimal AgentRunResult with state that has retries
        state = GraphAgentState()
        state.retries = 3  # Simulate 3 retries

        result = AgentRunResult(output="test", _state=state)

        # Use our extraction pattern
        state_obj = getattr(result, "_state", None)
        retries = getattr(state_obj, "retries", 0) if state_obj else 0

        assert retries == 3
