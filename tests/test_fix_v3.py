"""Unit tests for FIX v3:
(A) Lane labels as 5-6 word statements, no mid-phrase truncation.
(B) Lanes must never end blank (explicit lane states, exception isolation, gate parse retry).
"""
import pytest
from vresearch.agents import validate_lane_label, regenerate_lane_label, plan, RelevanceGateError
from vresearch.board import build_findings_board, _clean_lane_title
from vresearch.config import Settings
from vresearch.pipeline import run_research


# ==============================================================================
# PART A: Lane Label Validator & Regeneration Tests
# ==============================================================================

def test_lane_label_validator_accepts_valid_statements():
    valid_examples = [
        ("Opening lines and behavioral tactics", "What specific conversational and behavioral techniques do pickup artists use?"),
        ("Psychological manipulation techniques and effects", "How do pickup artists exploit psychological vulnerabilities in women?"),
        ("Key creators and online communities", "Who are the most influential creators, teachers, or communities that popularized these pickup artist techniques?"),
        ("Laws against deceptive dating tactics", "Are there any laws or regulatory actions against deceptive dating or pickup artist practices?"),
        ("Global battery demand and market growth", "How fast is EV battery demand growing?"),
    ]
    for label, question in valid_examples:
        ok, msg = validate_lane_label(label, question)
        assert ok, f"Expected '{label}' to be valid, but got error: {msg}"


def test_lane_label_validator_rejects_invalid_cases():
    question = "What specific conversational and behavioral techniques do pickup artists use to initiate and attract women?"

    # 1. Trailing question mark
    ok, _ = validate_lane_label("Who are the top creators?", question)
    assert not ok

    # 2. Ends on connective / stop word
    ok, _ = validate_lane_label("Opening lines and tactics of", question)
    assert not ok
    ok, _ = validate_lane_label("Opening lines and behavioral tactics and", question)
    assert not ok
    ok, _ = validate_lane_label("Tactics that are used for", question)
    assert not ok

    # 3. Verbatim prefix of question
    ok, _ = validate_lane_label("What specific conversational and behavioral", question)
    assert not ok

    # 4. Too short (< 5 words)
    ok, _ = validate_lane_label("Demand growth", question)
    assert not ok
    ok, _ = validate_lane_label("Core market drivers", question)
    assert not ok

    # 5. Too long (> 6 words)
    ok, _ = validate_lane_label("What are the most influential creators and teachers in history", question)
    assert not ok


def test_label_never_truncated_prefix_and_falls_back_to_full_question():
    """If the LLM provides an invalid label and regeneration fails, fallback uses the full question,
    never a truncated prefix or first N words."""
    class FailingLabelLLM:
        def json(self, system, user, *, model=None, tag="", max_tokens=0):
            if tag == "planner":
                return {
                    "sub_questions": [
                        {
                            "id": "Q1",
                            "label": "What specific conversational and behavioral",  # Invalid: verbatim prefix
                            "question": "What specific conversational and behavioral techniques do pickup artists use?",
                            "answer_spec": "Names specific tactics",
                            "queries": ["pua tactics"]
                        }
                    ]
                }
            if tag == "label_regen":
                # Regeneration also returns invalid
                return {"label": "Still too short"}
            return {}

    subqs = plan(FailingLabelLLM(), "test question")
    assert len(subqs) == 1
    # When regeneration fails, the label must equal the FULL question, NOT first N words
    assert subqs[0]["label"] == "What specific conversational and behavioral techniques do pickup artists use?"

    # Header title must never end mid-phrase
    title = _clean_lane_title("Q1", subqs[0]["label"], subqs[0]["question"])
    assert "and behavioral" not in title or "techniques" in title


# ==============================================================================
# PART B: Explicit Lane States & No Blank Lanes
# ==============================================================================

def test_lane_with_all_off_topic_claims_ends_in_empty_state_with_reason(tmp_path):
    """When all drafted claims are filtered as off-topic, the lane ends in state 'empty'
    with counts and reason; it is NEVER blank."""
    class AllOffTopicLLM:
        def json(self, system, user, *, model=None, tag="", max_tokens=0):
            if tag == "planner":
                return {"sub_questions": [
                    {
                        "id": "Q1",
                        "label": "Key creators and online communities",
                        "question": "Who are the most influential creators or communities?",
                        "answer_spec": "Names specific authors and communities",
                        "queries": ["pua gurus"]
                    }
                ]}
            if tag == "drafter":
                return {"claims": [
                    {"text": "Generic advice is to smile and speak clearly in social settings.", "evidence_ids": ["E1"]},
                    {"text": "General self-help books were popular in the 1990s decade.", "evidence_ids": ["E1"]}
                ]}
            if tag == "relevance":
                # Gate judges all as off-topic
                return {"evaluations": [
                    {"cid": "temp_Q1_1", "relevance": "off_topic", "reason": "generic conversational advice"},
                    {"cid": "temp_Q1_2", "relevance": "off_topic", "reason": "general self-help history"}
                ]}
            if tag == "researcher_refine":
                return {"queries": ["influential pua authors"]}
            if tag == "synthesizer":
                return {"summary": "", "sections": []}
            return {}

    def search_fn(q, n=6, **kwargs):
        return [{"title": "Self help", "url": "https://example.com/page", "snippet": "general self help advice"}]

    def fetch_fn(url):
        return "Generic advice is to smile and speak clearly in social settings. General self-help books were popular in the 1990s decade. Additional text for passage."

    s = Settings("k", "http://x", "m", "m", 1, 2, 4, 6, db_path=str(tmp_path / "test.db"))
    events_log = []

    def on_event(step, kind, payload):
        events_log.append({"step": step, "kind": kind, "payload": payload})

    run_id, rep = run_research(
        "Who are the most influential creators or communities?",
        settings=s,
        llm=AllOffTopicLLM(),
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        on_event=on_event
    )

    # Lane status events must exist
    lane_events = [e for e in events_log if e["step"] == "lane" and e["kind"] == "status"]
    assert len(lane_events) > 0

    # Final lane state for Q1 must be 'empty'
    final_q1_event = [e for e in lane_events if e["payload"]["subq"] == "Q1"][-1]
    assert final_q1_event["payload"]["state"] == "empty"
    assert "off-topic" in final_q1_event["payload"]["reason"].lower()
    assert final_q1_event["payload"]["counts"]["off_topic_dropped"] >= 2

    # Board must display the empty notice and reason, not be blank
    board = rep.get("board", {})
    lane = board["lanes"][0]
    assert lane["subq_id"] == "Q1"
    assert lane["is_thin"] is True
    assert lane["notes"] == []
    assert "off-topic" in lane["reason"].lower() or "not enough proof" in lane["thin_label"].lower()


def test_gate_parse_failure_retries_once_and_produces_error(tmp_path):
    """When relevance gate JSON fails to parse on both initial call and retry,
    the lane transitions to state 'error', not silent drop or silent keep."""
    class GateFailingLLM:
        relevance_calls = 0

        def json(self, system, user, *, model=None, tag="", max_tokens=0):
            if tag == "planner":
                return {"sub_questions": [
                    {
                        "id": "Q1",
                        "label": "Key creators and online communities",
                        "question": "Who are the most influential creators or communities?",
                        "answer_spec": "Names specific authors",
                        "queries": ["pua gurus"]
                    }
                ]}
            if tag == "drafter":
                return {"claims": [
                    {"text": "Mystery established Project Hollywood in California.", "evidence_ids": ["E1"]}
                ]}
            if tag == "relevance":
                self.relevance_calls += 1
                # Simulates JSON decode / parse error
                raise ValueError("Malformed JSON string from LLM")
            if tag == "synthesizer":
                return {"summary": "", "sections": []}
            return {}

    def search_fn(q, n=6, **kwargs):
        return [{"title": "Page", "url": "https://example.com/p", "snippet": "Mystery established Project Hollywood."}]

    def fetch_fn(url):
        return "Mystery established Project Hollywood in California. Seduction community history."

    llm = GateFailingLLM()
    s = Settings("k", "http://x", "m", "m", 0, 1, 4, 6, db_path=str(tmp_path / "test.db"))
    events_log = []

    def on_event(step, kind, payload):
        events_log.append({"step": step, "kind": kind, "payload": payload})

    run_id, rep = run_research(
        "Who are the most influential creators?",
        settings=s,
        llm=llm,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        on_event=on_event
    )

    # Assert retry occurred: 2 calls to relevance gate
    assert llm.relevance_calls == 2, "Relevance gate must retry once on parse failure!"

    # Lane status must be 'error'
    lane_events = [e for e in events_log if e["step"] == "lane" and e["kind"] == "status" and e["payload"]["subq"] == "Q1"]
    assert any(e["payload"]["state"] == "error" for e in lane_events)
    err_event = [e for e in lane_events if e["payload"]["state"] == "error"][0]
    assert "parse failure" in err_event["payload"]["reason"].lower() or "error" in err_event["payload"]["reason"].lower()


def test_worker_exception_in_one_lane_yields_lane_status_error_and_others_finish(tmp_path):
    """When a worker in one lane throws an unhandled exception, it emits lane/status error
    and other lanes complete normally."""
    class WorkerExceptionLLM:
        def json(self, system, user, *, model=None, tag="", max_tokens=0):
            if tag == "planner":
                return {"sub_questions": [
                    {
                        "id": "Q1",
                        "label": "Failing lane with worker error",
                        "question": "What happens in failing lane?",
                        "answer_spec": "Facts",
                        "queries": ["fail query"]
                    },
                    {
                        "id": "Q2",
                        "label": "Healthy lane that succeeds normally",
                        "question": "What happens in healthy lane?",
                        "answer_spec": "Facts",
                        "queries": ["healthy query"]
                    }
                ]}
            if tag == "drafter":
                import re
                ids = re.findall(r"^\[(E\d+)\]", user, re.M)
                e = ids[0] if ids else "E1"
                return {"claims": [
                    {"text": "Healthy claim is verified and accepted.", "evidence_ids": [e]}
                ]}
            if tag == "relevance":
                return {"evaluations": [{"cid": "temp_Q2_1", "relevance": "direct", "reason": "direct answer"}]}
            if tag == "verifier":
                return {"verdict": "supported", "quote": "Healthy claim is verified and accepted.", "reason": "stated verbatim"}
            if tag == "synthesizer":
                return {"summary": "Healthy claim is verified and accepted [C1].", "sections": [{"id": "Q2", "text": "Healthy claim [C1]."}]}
            return {}

    def search_fn(q, n=6, **kwargs):
        if "fail query" in q:
            raise RuntimeError("Simulated unhandled worker crash in research worker!")
        return [{"title": "Page", "url": "https://example.com/p", "snippet": "Healthy claim is verified and accepted."}]

    def fetch_fn(url):
        return "Healthy claim is verified and accepted. Substantive evidence text."

    s = Settings("k", "http://x", "m", "m", 0, 1, 4, 6, db_path=str(tmp_path / "test.db"))
    events_log = []

    def on_event(step, kind, payload):
        events_log.append({"step": step, "kind": kind, "payload": payload})

    run_id, rep = run_research(
        "Test worker exception handling across parallel lanes",
        settings=s,
        llm=WorkerExceptionLLM(),
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        on_event=on_event
    )

    # Q1 must have emitted lane/status error
    q1_events = [e for e in events_log if e["step"] == "lane" and e["kind"] == "status" and e["payload"]["subq"] == "Q1"]
    assert any(e["payload"]["state"] == "error" for e in q1_events)
    assert any("worker error" in e["payload"]["reason"].lower() for e in q1_events)

    # Q2 must have completed successfully and verified claims
    q2_events = [e for e in events_log if e["step"] == "lane" and e["kind"] == "status" and e["payload"]["subq"] == "Q2"]
    assert any(e["payload"]["state"] == "done" for e in q2_events)
    assert len(rep["accepted"]) >= 1
    assert "Healthy claim" in rep["accepted"][0]["text"]
