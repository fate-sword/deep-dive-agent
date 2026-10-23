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


def test_provider_registry_only_two_providers():
    # Only mesh and openrouter
    p_mesh = get_provider("mesh")
    assert p_mesh.name == "mesh"
    assert p_mesh.dev_only is False

    with pytest.raises(ValueError, match="Deeptrace registry strictly supports 'mesh' and 'openrouter'"):
        get_provider("openai")

    with pytest.raises(ValueError, match="Deeptrace registry strictly supports 'mesh' and 'openrouter'"):
        get_provider("anthropic")


def test_openrouter_guardrails(monkeypatch):
    # Refuse if ALLOW_DEV_PROVIDER is not set
    monkeypatch.delenv("ALLOW_DEV_PROVIDER", raising=False)
    with pytest.raises(RuntimeError, match="requires ALLOW_DEV_PROVIDER=1"):
        get_provider("openrouter")

    # Refuse in production deploy
    monkeypatch.setenv("ALLOW_DEV_PROVIDER", "1")
    monkeypatch.setenv("DEPLOY_ENV", "production")
    with pytest.raises(RuntimeError, match="Deploy environment is production"):
        get_provider("openrouter")


def test_role_based_model_resolution(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("ALLOW_DEV_PROVIDER", "1")
    monkeypatch.delenv("DEPLOY_ENV", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", "default/model:free")
    monkeypatch.setenv("OPENROUTER_MODEL_VERIFIER", "verifier/special:free")

    s = get_settings()
    assert s.get_model_for_role("planner") == "default/model:free"
    assert s.get_model_for_role("drafter") == "default/model:free"
    assert s.get_model_for_role("verifier") == "verifier/special:free"
    assert s.get_model_for_tag("contradictor") == "verifier/special:free"
    assert s.get_model_for_tag("synthesizer") == "default/model:free"


def test_extra_body_json_validation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mesh")
    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", '{"reasoning": {"effort": "none"}}')
    s = get_settings()
    assert s.llm_extra_body == {"reasoning": {"effort": "none"}}

    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", 'invalid-json')
    with pytest.raises(ValueError, match="Invalid LLM_EXTRA_BODY_JSON"):
        get_settings()
