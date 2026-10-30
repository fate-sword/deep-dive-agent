"""Tests for provider abstraction, role routing, reasoning stripping, and guardrails."""
import json
import os
import pytest

from vresearch.config import get_settings, map_tag_to_role, Settings
from vresearch.llm import extract_json, strip_thinking
from vresearch.providers import get_provider, Provider


def test_strip_thinking_tags():
    assert strip_thinking("Hello <think>internal reasoning</think> world") == "Hello  world"
    multiline = "<think>\nThinking step 1\nThinking step 2\n</think>Answer is 42."
    assert strip_thinking(multiline) == "Answer is 42."
    case_insensitive = "<THINK>thoughts</THINK>Valid output"
    assert strip_thinking(case_insensitive) == "Valid output"


def test_extract_json_with_thinking_blocks():
    raw = "<think>Let me reason.\nI will produce JSON.\n</think>\n```json\n{\"verdict\": \"supported\", \"quote\": \"abc\"}\n```"
    data = extract_json(raw)
    assert data == {"verdict": "supported", "quote": "abc"}

    raw_raw = "<think>some reasoning</think>{\"key\": [1, 2, 3]}"
    assert extract_json(raw_raw) == {"key": [1, 2, 3]}


def test_map_tag_to_role():
    assert map_tag_to_role("planner") == "planner"
    assert map_tag_to_role("researcher_refine") == "planner"
    assert map_tag_to_role("drafter") == "drafter"
    assert map_tag_to_role("verifier") == "verifier"
    assert map_tag_to_role("contradictor") == "verifier"
    assert map_tag_to_role("synthesizer") == "synthesizer"
    assert map_tag_to_role("followup") == "synthesizer"
    assert map_tag_to_role("unknown_tag") == "planner"


def test_provider_registry_mesh_only():
    p_mesh = get_provider("mesh")
    assert p_mesh.name == "mesh"
    assert p_mesh.dev_only is False

    with pytest.raises(ValueError, match="Deeptrace registry strictly supports 'mesh'"):
        get_provider("openai")

    with pytest.raises(ValueError, match="Deeptrace registry strictly supports 'mesh'"):
        get_provider("anthropic")


def test_role_based_model_resolution(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mesh")
    monkeypatch.delenv("DEPLOY_ENV", raising=False)
    monkeypatch.setenv("MESH_MODEL", "default/model")
    monkeypatch.setenv("MESH_VERIFIER_MODEL", "verifier/special")

    s = get_settings()
    assert s.get_model_for_role("planner") == "default/model"
    assert s.get_model_for_role("drafter") == "default/model"
    assert s.get_model_for_role("verifier") == "verifier/special"
    assert s.get_model_for_tag("contradictor") == "verifier/special"
    assert s.get_model_for_tag("synthesizer") == "default/model"


def test_extra_body_json_validation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mesh")
    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", '{"reasoning": {"effort": "none"}}')
    s = get_settings()
    assert s.llm_extra_body == {"reasoning": {"effort": "none"}}

    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", 'invalid-json')
    with pytest.raises(ValueError, match="Invalid LLM_EXTRA_BODY_JSON"):
        get_settings()


def test_synthesizer_max_tokens_and_reasoning_effort():
    from unittest.mock import MagicMock
    import vresearch.agents as agents
    from vresearch.llm import LLM

    s = Settings("mock-key", "https://api.meshapi.ai/v1", "z-ai/glm-5.3-flash", "z-ai/glm-5.3-flash", 2, 2, 4, 6, ":memory:", llm_provider="mesh")
    llm = LLM(settings=s, provider="mesh")
    captured_kwargs = []

    def mock_create(**kwargs):
        captured_kwargs.append(kwargs)
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = '{"summary": "Test summary [C1].", "sections": []}'
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.total_tokens = 100
        mock_resp.model = "z-ai/glm-5.3-flash"
        return mock_resp

    llm.client.chat.completions.create = mock_create

    res = agents.synthesize(llm, "What is the test?", [{"id": "Q1", "question": "Sub Q?"}], [{"cid": "C1", "subq_id": "Q1", "text": "Claim 1", "status": "verified"}], [], set())
    assert res == {"summary": "Test summary [C1].", "sections": []}
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["max_tokens"] == 4500
    assert captured_kwargs[0]["extra_body"] == {"reasoning_effort": "low"}


def test_empty_output_length_retry():
    from unittest.mock import MagicMock
    from vresearch.llm import LLM

    s = Settings("mock-key", "https://api.meshapi.ai/v1", "z-ai/glm-5.3-flash", "z-ai/glm-5.3-flash", 2, 2, 4, 6, ":memory:", llm_provider="mesh")
    llm = LLM(settings=s, provider="mesh")
    calls = []
    llm.on_llm_call = lambda d: calls.append(d)
    captured_kwargs = []

    call_count = 0
    def mock_create(**kwargs):
        nonlocal call_count
        captured_kwargs.append(kwargs)
        call_count += 1
        mock_choice = MagicMock()
        mock_resp = MagicMock()
        mock_resp.usage.total_tokens = 2500
        mock_resp.model = "z-ai/glm-5.3-flash"
        if call_count == 1:
            mock_choice.finish_reason = "length"
            mock_choice.message.content = ""
        else:
            mock_choice.finish_reason = "stop"
            mock_choice.message.content = '{"summary": "Recovered summary [C1].", "sections": []}'
        mock_resp.choices = [mock_choice]
        return mock_resp

    llm.client.chat.completions.create = mock_create

    out = llm.json("system prompt", "user query", tag="synthesizer", max_tokens=2500)
    assert out["summary"] == "Recovered summary [C1]."
    assert len(captured_kwargs) == 2
    assert captured_kwargs[0]["max_tokens"] == 2500
    assert captured_kwargs[1]["max_tokens"] > 2500
    assert len(calls) == 2
    assert calls[0]["tag"] == "synthesizer"
    assert "reason" not in calls[0]
    assert calls[1]["tag"] == "synthesizer:retry"
    assert calls[1]["reason"] == "empty output"

