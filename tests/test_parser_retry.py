"""Unit tests for Fix 2: JSON extraction and retry handling.

Asserts:
1. Truncated repro does NOT return a partial inner object and raises ValueError.
2. Valid fenced JSON parses successfully.
3. Prose-wrapped JSON parses successfully.
4. Top-level JSON array parses successfully.
"""
import pytest
from vresearch.llm import extract_json


def test_truncated_repro_does_not_return_partial_inner_object():
    """When JSON output is truncated midway (e.g. hitting max_tokens),
    extract_json must raise ValueError, NEVER returning a partial inner dict.
    """
    truncated_raw = '{"evaluations": [{"cid": "temp_Q2_1", "relevance": "direct", "reason": "test"}, {"cid": "temp_'
    with pytest.raises(ValueError, match=r"no valid JSON|truncated"):
        extract_json(truncated_raw)


def test_valid_fenced_json():
    """Code-fenced JSON ```json ... ``` is parsed cleanly."""
    fenced_raw = '```json\n{"evaluations": [{"cid": "temp_Q2_1", "relevance": "direct", "reason": "test"}]}\n```'
    parsed = extract_json(fenced_raw)
    assert isinstance(parsed, dict)
    assert "evaluations" in parsed
    assert len(parsed["evaluations"]) == 1
    assert parsed["evaluations"][0]["cid"] == "temp_Q2_1"


def test_prose_wrapped_json():
    """JSON wrapped with conversational prose before and after is extracted correctly."""
    prose_raw = (
        "Here is the evaluation for the claims:\n\n"
        '{"evaluations": [{"cid": "temp_Q2_1", "relevance": "direct", "reason": "test"}]}\n\n'
        "Hope this helps!"
    )
    parsed = extract_json(prose_raw)
    assert isinstance(parsed, dict)
    assert "evaluations" in parsed
    assert len(parsed["evaluations"]) == 1
    assert parsed["evaluations"][0]["cid"] == "temp_Q2_1"


def test_top_level_array():
    """A top-level JSON array is extracted and supported."""
    array_raw = '[{"cid": "temp_Q2_1", "relevance": "direct", "reason": "test"}]'
    parsed = extract_json(array_raw)
    assert isinstance(parsed, (list, dict))
    items = parsed["evaluations"] if isinstance(parsed, dict) else parsed
    assert len(items) == 1
    assert items[0]["cid"] == "temp_Q2_1"
