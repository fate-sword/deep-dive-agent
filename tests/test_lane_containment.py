"""Test for Item 1b: Lane error containment.

Asserts:
- When one lane's relevance gate fails, NO event with kind=='error' is emitted.
- Other lanes continue drafting and verifying normally.
- Coverage node loops back for the failed lane.
- Final report event / report object is generated and emitted.
"""
import pytest
from vresearch.agents import RelevanceGateError
from vresearch.config import Settings
from vresearch.pipeline import run_research


class GateFailureOneLaneLLM:
    """Fake LLM where lane Q2 fails relevance gate in loop 0, but Q1 succeeds,
    and Q2 recovers on loopback."""
    def __init__(self):
        self.loop = 0
        self.relevance_q2_calls = 0

    def json(self, system, user, *, model=None, tag="", max_tokens=0):
        if tag == "planner":
            return {"sub_questions": [
                {
                    "id": "Q1",
                    "label": "Healthy lane topic one",
                    "question": "What is verified in healthy lane?",
                    "answer_spec": "Empirical facts",
                    "queries": ["healthy query"]
                },
                {
                    "id": "Q2",
                    "label": "Failing lane topic two",
                    "question": "What happens in failing lane?",
                    "answer_spec": "Empirical facts",
                    "queries": ["failing query"]
                }
            ]}

        if tag == "researcher_refine":
            return {"queries": ["retry query for failing lane"]}

        if tag == "drafter":
            if "failing" in user.lower():
                return {"claims": [
                    {"text": "Failing lane claim to be tested.", "evidence_ids": ["E2"]}
                ]}
            return {"claims": [
                {"text": "Healthy claim verified and cited.", "evidence_ids": ["E1"]}
            ]}

        if tag == "relevance":
            if "failing" in user.lower():
                self.relevance_q2_calls += 1
                if self.relevance_q2_calls <= 2:
                    # Simulates relevance gate failure on loop 0 (both attempts)
                    raise ValueError("Simulated gate parse failure on Q2")
                # Loop 1 recovery
                return {"evaluations": [
                    {"cid": "temp_Q2_1", "relevance": "direct", "reason": "recovered in loop 1"}
                ]}
            # Q1 succeeds
            return {"evaluations": [
                {"cid": "temp_Q1_1", "relevance": "direct", "reason": "direct finding"}
            ]}

        if tag == "verifier":
            return {
                "verdict": "supported",
                "quote": "verified and cited verbatim",
                "reason": "stated in passage"
            }

        if tag == "synthesizer":
            return {
                "summary": "Synthesized report summarizing findings [C1].",
                "sections": [{"id": "Q1", "text": "Section content [C1]."}]
            }

        return {}


def test_lane_gate_failure_emits_no_error_event_and_completes_run(tmp_path):
    def search_fn(q, n=6, **kwargs):
        if "healthy" in q:
            return [{"title": "Healthy Page", "url": "https://example.com/h", "snippet": "Healthy claim verified and cited."}]
        return [{"title": "Failing Page", "url": "https://example.com/f", "snippet": "Failing lane claim to be tested."}]

    def fetch_fn(url):
        if "example.com/h" in url:
            return "Healthy claim verified and cited verbatim in source text."
        return "Failing lane claim to be tested in source text."

    llm = GateFailureOneLaneLLM()
    s = Settings("k", "http://x", "m", "m", max_loops=1, min_verified_per_subq=1, pages_per_subq=4, results_per_query=6, db_path=str(tmp_path / "test.db"))
    events_log = []

    def on_event(step, kind, payload):
        events_log.append({"step": step, "kind": kind, "payload": payload})

    run_id, rep = run_research(
        "Test lane failure containment across parallel pipeline",
        settings=s,
        llm=llm,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        on_event=on_event
    )

    # 1. Assert NO event with kind == 'error' was emitted!
    error_events = [e for e in events_log if e["kind"] == "error"]
    assert len(error_events) == 0, f"Expected 0 kind='error' events, got: {error_events}"

    # 2. Assert lane_warning was emitted for Q2
    warning_events = [e for e in events_log if e["step"] == "draft" and e["kind"] == "lane_warning"]
    assert len(warning_events) >= 1
    assert warning_events[0]["payload"]["subq"] == "Q2"
    assert "error" in warning_events[0]["payload"]
    assert "message" in warning_events[0]["payload"]

    # 3. Assert Q1 verified normally
    q1_verdicts = [e for e in events_log if e["step"] == "verify" and e["kind"] == "verdict" and e["payload"]["cid"] == "C1"]
    assert len(q1_verdicts) == 1
    assert q1_verdicts[0]["payload"]["status"] == "verified"

    # 4. Assert coverage looped back for failed lane Q2
    loop_events = [e for e in events_log if e["step"] == "coverage" and e["kind"] == "loop_back"]
    assert len(loop_events) == 1
    assert "Q2" in loop_events[0]["payload"]["weak"]

    # 5. Assert final report was generated and returned
    assert rep is not None
    assert "summary" in rep
    assert len(rep.get("accepted", [])) >= 1
