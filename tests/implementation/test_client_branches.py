"""Additional branch coverage for client orchestration.

Doctest summary:
>>> models = ["m1", "m2", "m1"]
>>> len(models)
3
"""

from unittest.mock import patch

import pytest

from covenance import Covenance
from covenance._backend_result import BackendResult
from covenance.exceptions import StructuredOutputParsingError
from covenance.record import TokenUsage


def _usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)


def test_call_native_provider_rejects_unknown_provider():
    client = Covenance()
    with pytest.raises(ValueError, match="No native backend for provider"):
        client._call_native_provider(
            user_msg="u",
            model="m",
            provider="unknown",
            llm_type=None,
            sys_msg=None,
            temperature=None,
        )


def test_ask_native_validation_error_becomes_structured_parsing_error():
    client = Covenance()

    with patch.object(client, "_call_native_provider") as mock_call:
        mock_call.return_value = BackendResult(output="not-an-int", usage=_usage())
        with pytest.raises(StructuredOutputParsingError):
            client._ask_native(
                user_msg="u",
                model="gpt-4o",
                provider="openai",
                response_type=int,
                sys_msg=None,
                max_parsing_retries=0,
                temperature=None,
            )


def test_llm_consensus_with_single_candidate_delegates_to_ask_llm():
    client = Covenance()

    with patch.object(client, "ask_llm", return_value="single") as mock_ask:
        result = client.llm_consensus(
            user_msg="question",
            model="gpt-4o",
            num_candidates=1,
        )

    assert result == "single"
    mock_ask.assert_called_once()


def test_llm_consensus_parallel_formats_dict_list_and_plain_candidates():
    client = Covenance()
    call_counter = {"n": 0}
    integration_payload = {"value": "integrated"}

    def fake_ask_llm(**kwargs):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            return {"a": 1}
        if call_counter["n"] == 2:
            return ["x", 2]
        if call_counter["n"] == 3:
            return "plain-candidate"
        return integration_payload

    with patch.object(client, "ask_llm", side_effect=fake_ask_llm) as mock_ask:
        result = client.llm_consensus(
            user_msg="Integrate these answers",
            model="gpt-4o",
            num_candidates=3,
            additional_models=["m1", "m2"],
            parallel=True,
        )

    assert result == integration_payload
    assert mock_ask.call_count == 4

    # Last call is integration.
    integration_kwargs = mock_ask.call_args_list[-1].kwargs
    text = integration_kwargs["user_msg"]
    assert "Candidate Answer 1" in text
    assert "Candidate Answer 2" in text
    assert "Candidate Answer 3" in text
    assert '"a": 1' in text
    assert "plain-candidate" in text

    worker_models = [c.kwargs["model"] for c in mock_ask.call_args_list[:3]]
    assert sorted(worker_models) == ["m1", "m1", "m2"]


def test_usage_summary_and_print_helpers_delegate_to_record_and_visual_modules():
    client = Covenance(label="branch-test")

    with patch("covenance.record.usage_summary", return_value={"calls": 0}) as summary:
        result = client.usage_summary()
        assert result == {"calls": 0}
        summary.assert_called_once()

    with patch("covenance.record.print_usage") as print_usage:
        client.print_usage()
        kwargs = print_usage.call_args.kwargs
        assert kwargs["title"] == "LLM Usage Summary (branch-test)"

    with patch("covenance.visual.print_call_timeline") as print_timeline:
        client.print_call_timeline(width=120)
        kwargs = print_timeline.call_args.kwargs
        assert kwargs["width"] == 120
        assert isinstance(kwargs["records"], list)
