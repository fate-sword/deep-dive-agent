"""Unit tests for Fix v2: Claim relevance gate, passage relevance filter, and generic claim suppression.

Asserts:
- Generic / off-topic claims are dropped by the gate and never appear in cards, verify events, report, board, or exports.
- A lane with only generic evidence loops back.
- Dropped claims appear exclusively in the Decision Trail filtered section.
- NO draft/claims or verify/verdict event ever carries an off_topic claim.
- Relevant claims are judged direct, verified, and shown.
- Relevance rate metric is tracked accurately.
"""
import re
import pytest
from vresearch.agents import evaluate_claim_relevance
from vresearch.config import Settings
from vresearch.export import to_html, to_markdown
from vresearch.pipeline import run_research


class GenericVsDirectFakeLLM:
    """Fake LLM simulating generic background warnings vs direct specific evidence."""
    calls = 0

    def json(self, system, user, *, model=None, tag="", max_tokens=0):
        self.calls += 1
        if tag == "planner":
            return {"sub_questions": [
                {
                    "id": "Q1",
                    "label": "weight loss supplements",
                    "question": "What are evidence-based natural supplements that support weight loss?",
                    "answer_spec": "names specific supplements or ingredients, each with trial evidence on efficacy and safety",
                    "queries": ["natural supplements weight loss trials"]
                }
            ]}
        if tag == "researcher_refine":
            return {"queries": ["green tea extract glucomannan weight loss clinical trials"]}
        if tag == "drafter":
            ids = re.findall(r"^\[(E\d+)\]", user, re.M)
            e = ids[0] if ids else "E1"
            if "yellow oleander" in user.lower() and "green tea" not in user.lower():
                return {"claims": [
                    {"text": "Yellow oleander is a poisonous plant and a toxic public health hazard.", "evidence_ids": [e]}
                ]}
            return {"claims": [
                {"text": "Green tea extract resulted in significant weight loss in clinical trials.", "evidence_ids": [e]},
                {"text": "Glucomannan supplementation produced modest body weight reductions.", "evidence_ids": [e]}
            ]}
        if tag == "relevance":
            # Relevance Gate: parse each line and judge direct vs off_topic
            evals = []
            for line in user.splitlines():
                m = re.match(r"^\[(temp_[^\]]+)\]\s*(.*)", line.strip())
                if m:
                    cid, text = m.group(1), m.group(2)
                    if "yellow oleander" in text.lower():
                        evals.append({
                            "cid": cid,
                            "relevance": "off_topic",
                            "reason": "warning about toxic plant does not name a safe weight loss supplement"
                        })
                    else:
                        evals.append({
                            "cid": cid,
                            "relevance": "direct",
                            "reason": "names specific evidence-based supplement with trial results"
                        })
            return {"evaluations": evals}
        if tag == "verifier":
            quote = ""
            if "green tea" in user.lower():
                quote = "Green tea extract resulted in significant weight loss in clinical trials."
            elif "glucomannan" in user.lower():
                quote = "Glucomannan supplementation produced modest body weight reductions."
            return {
                "verdict": "supported",
                "quote": quote,
                "reason": "stated verbatim in source passage"
            }
        if tag == "synthesizer":
            return {
                "summary": "Clinical trials demonstrate that green tea extract and glucomannan aid weight loss [C1][C2].",
                "sections": [{"id": "Q1", "text": "Green tea extract showed measurable efficacy [C1]. Glucomannan reduced body weight [C2]."}]
            }
        raise AssertionError(f"Unexpected tag: {tag}")


def test_generic_passages_dropped_by_gate_and_lane_loops_back(tmp_path):
    """When retrieval returns generic warnings, the gate drops claims, the lane loops back,
    and NO generic claim enters report, board, cards, or exports.
    """
    round_count = 0

    def search_fn(query, n=6, **kwargs):
        nonlocal round_count
        round_count += 1
        if round_count == 1:
            # Round 0: generic FDA warning only
            return [{"title": "FDA Warning", "url": "https://fda.gov/warning-yellow-oleander", "snippet": "toxic yellow oleander"}]
        # Loop-back round: returns relevant green tea study
        return [{"title": "Clinical Trial", "url": "https://pubmed.ncbi.nlm.nih.gov/12345", "snippet": "green tea extract trial"}]

    def fetch_fn(url):
        if "fda.gov" in url:
            return "Yellow oleander is a poisonous plant and a toxic public health hazard."
        return (
            "Green tea extract resulted in significant weight loss in clinical trials. "
            "Glucomannan supplementation produced modest body weight reductions. "
            "Evidence shows both natural supplements support weight reduction in clinical research."
        )

    s = Settings("k", "http://x", "m", "m", max_loops=2, min_verified_per_subq=2, pages_per_subq=4, results_per_query=6, db_path=str(tmp_path / "test.db"))
    llm = GenericVsDirectFakeLLM()
    events_log = []

    def on_event(step, kind, payload):
        events_log.append({"step": step, "kind": kind, "payload": payload})

    run_id, rep = run_research(
        "What are evidence-based natural supplements that support weight loss?",
        settings=s,
        llm=llm,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        on_event=on_event
    )

    # 1. Assert loop-back happened
    assert rep["metrics"]["loops"] >= 1, "Lane with generic evidence should have looped back!"
    loop_events = [e for e in events_log if e["step"] == "coverage" and e["kind"] == "loop_back"]
    assert len(loop_events) >= 1

    # 2. Assert NO draft/claims or verify/verdict ever carried an off_topic claim
    for ev in events_log:
        if ev["step"] == "draft" and ev["kind"] == "claims":
            for itm in ev["payload"].get("items", []):
                assert "yellow oleander" not in itm.get("text", "").lower(), "draft/claims must NEVER contain an off-topic claim!"
        if ev["step"] == "verify" and ev["kind"] == "verdict":
            assert "yellow oleander" not in ev["payload"].get("claim", "").lower(), "verify/verdict must NEVER verify an off-topic claim!"

    # 3. Assert yellow oleander NEVER appears in accepted, unverified, or rejected
    all_final_claims = rep["accepted"] + rep["unverified"] + rep["rejected"]
    for c in all_final_claims:
        assert "yellow oleander" not in c["text"].lower()

    # 4. Assert yellow oleander does NOT appear in Findings Board notes
    board = rep.get("board", {})
    for lane in board.get("lanes", []):
        for note in lane.get("notes", []):
            assert "yellow oleander" not in note["text"].lower()

    # 5. Assert yellow oleander does NOT appear in Markdown or HTML exports
    md = to_markdown(rep)
    assert "yellow oleander" not in md.lower()
    html = to_html(rep)
    assert "yellow oleander" not in html.lower()

    # 6. Assert dropped claim IS stored in Decision Trail filtered section
    from vresearch.store import RunStore
    store = RunStore(str(tmp_path / "test.db"))
    filtered = store.load_filtered_claims(run_id)
    assert len(filtered) >= 1
    assert "yellow oleander" in filtered[0]["text"].lower()
    assert "warning" in filtered[0]["reason"].lower() or "off_topic" in filtered[0]["reason"].lower()

    # 7. Assert relevant claims are accepted
    accepted_texts = {c["text"] for c in rep["accepted"]}
    assert any("green tea" in t.lower() for t in accepted_texts)


def test_relevance_rate_metric_and_specifics_eval(tmp_path):
    """Asserts that the relevance_rate metric is tracked correctly and records evaluation numbers."""
    class EvaluatorLLM:
        def json(self, system, user, *, model=None, tag="", max_tokens=0):
            if tag == "planner":
                return {"sub_questions": [
                    {"id": "Q1", "label": "supplements", "question": "which named supplements have trial evidence for weight loss", "answer_spec": "names specific supplements with trial results", "queries": ["weight loss supplements trials"]}
                ]}
            if tag == "drafter":
                ids = re.findall(r"^\[(E\d+)\]", user, re.M)
                e = ids[0] if ids else "E1"
                return {"claims": [
                    {"text": "Green tea extract increased fat oxidation in clinical trials.", "evidence_ids": [e]},
                    {"text": "The earth revolves around the sun once every 365 days.", "evidence_ids": [e]}
                ]}
            if tag == "relevance":
                evals = []
                for line in user.splitlines():
                    m = re.match(r"^\[(temp_[^\]]+)\]\s*(.*)", line.strip())
                    if m:
                        cid, text = m.group(1), m.group(2)
                        if "green tea" in text.lower():
                            evals.append({"cid": cid, "relevance": "direct", "reason": "names specific supplement"})
                        else:
                            evals.append({"cid": cid, "relevance": "off_topic", "reason": "astronomy fact unrelated to supplements"})
                return {"evaluations": evals}
            if tag == "verifier":
                return {"verdict": "supported", "quote": "Green tea extract increased fat oxidation in clinical trials.", "reason": "stated in source"}
            if tag == "synthesizer":
                return {"summary": "Green tea extract has trial evidence for weight loss [C1].", "sections": [{"id": "Q1", "text": "Green tea extract is effective [C1]."}]}
            return {}

    def search_fn(q, n=6, **kwargs):
        return [{"title": "Trial", "url": "https://trials.org/tea", "snippet": "green tea trial"}]

    def fetch_fn(url):
        return (
            "Green tea extract increased fat oxidation in clinical trials. "
            "The earth revolves around the sun once every 365 days. "
            "Additional clinical evidence demonstrates weight management benefits."
        )

    s = Settings("k", "http://x", "m", "m", max_loops=1, min_verified_per_subq=1, pages_per_subq=4, results_per_query=6, db_path=str(tmp_path / "eval.db"))
    llm = EvaluatorLLM()

    run_id, rep = run_research(
        "which named supplements have trial evidence for weight loss",
        settings=s,
        llm=llm,
        search_fn=search_fn,
        fetch_fn=fetch_fn
    )

    m = rep["metrics"]
    assert m["claims_drafted"] == 2
    assert m["claims_filtered_off_topic"] == 1
    assert m["claims_total"] == 1  # 1 kept
    assert m["relevance_rate"] == 0.5  # 1 / 2 = 0.50

    # Ensure off-topic astronomy fact is not in accepted claims
    accepted_texts = [c["text"] for c in rep["accepted"]]
    assert len(accepted_texts) == 1
    assert "Green tea extract" in accepted_texts[0]
    assert "earth revolves" not in accepted_texts[0]
