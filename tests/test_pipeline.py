"""Offline end-to-end tests: a scripted fake LLM + fake search exercise the real LangGraph pipeline,
including the loop-back, claim rejection, programmatic grounding checks, contradictions and exports."""
import re

import pytest

from vresearch import pipeline
from vresearch.agents import _norm
from vresearch.config import Settings
from vresearch.export import to_html, to_markdown, to_pdf
from vresearch.llm import extract_json
from vresearch.pipeline import answer_followup, run_research

PAGE_A = ("Battery pack prices fell 20% in 2024 according to the market survey published this year.\n"
          "Lithium carbonate prices dropped sharply, which reduced cell costs for manufacturers worldwide.\n"
          "Analysts expect oversupply of cells to continue pressuring margins across the industry.\n"
          "Several producers announced capacity cuts while demand growth slowed in some regions this quarter.")
PAGE_B = ("Regulators in the region introduced new recycling rules that raise compliance costs for makers.\n"
          "The rules take effect next year and apply to all battery producers selling in the market.\n"
          "Companies that fail to comply face fines and possible loss of market access.\n"
          "Industry groups asked for a longer transition period but regulators have not agreed so far.")


class FakeLLM:
    calls = 0

    def json(self, system, user, *, model=None, tag="", max_tokens=0):
        self.calls += 1
        if tag == "planner":
            return {"sub_questions": [
                {"id": "Q1", "question": "What is happening to prices?", "queries": ["alpha prices"]},
                {"id": "Q2", "question": "What regulation applies?", "queries": ["beta regulation"]}]}
        if tag == "researcher_refine":
            return {"queries": ["gamma regulation rules"]}
        if tag == "drafter":
            ids = re.findall(r"^\[(E\d+)\]", user, re.M)
            if "prices" in user.split("\n")[0]:
                e = ids[0]
                return {"claims": [
                    {"text": "Battery pack prices fell 20% in 2024.", "evidence_ids": [e]},
                    {"text": "Lithium carbonate prices dropped sharply.", "evidence_ids": [e]},
                    {"text": "Sales tripled by 2030.", "evidence_ids": [e]},           # hallucination -> rejected
                    {"text": "Pack prices fell 45% in 2024.", "evidence_ids": [e]},    # number not in source
                    {"text": "Cells are cheap. invented quote", "evidence_ids": [e]},  # quote not verbatim
                    {"text": "Totally uncited claim.", "evidence_ids": ["E999"]}]}    # bad id -> no source
            return {"claims": [{"text": "New recycling rules raise compliance costs.", "evidence_ids": [ids[0]]},
                               {"text": "Non-compliant firms face fines.", "evidence_ids": [ids[0]]}]}
        if tag == "verifier":
            claim = re.search(r"CLAIM: (.*)", user).group(1)
            passage = re.search(r"\[E\d+\] (.{60})", user).group(1)
            if "tripled" in claim:
                return {"verdict": "unsupported", "quote": "", "reason": "source says nothing about sales"}
            if "invented quote" in claim:
                return {"verdict": "supported", "quote": "this text is not in the source", "reason": "ok"}
            return {"verdict": "supported", "quote": passage, "reason": "stated in source"}
        if tag == "contradictor":
            ids = re.findall(r"^(C\d+):", user, re.M)
            return {"contradictions": [{"claim_ids": ids[:2], "explanation": "Two figures conflict."}]}
        if tag == "synthesizer":
            sec = {}
            cur = None
            for line in user.splitlines():
                m = re.match(r"(Q\d):", line)
                if m:
                    cur = m.group(1)
                m = re.match(r"\s+(C\d+)", line)
                if m and cur and cur not in sec:
                    sec[cur] = m.group(1)
            return {"summary": "Prices are falling [C1]. This is obviously great.",
                    "sections": [{"id": q, "text": f"Finding [{c}]. Bogus [C99]. No cite here."} for q, c in sec.items()]}
        if tag == "followup":
            return {"answer": "Prices fell in 2024 [C1]. Unsupported aside."}
        raise AssertionError(tag)


def fake_search(query, n=6):
    if "alpha" in query:
        return [{"title": "Survey", "url": "https://www.reuters.com/a", "snippet": "s"}]
    if "gamma" in query:
        return [{"title": "Rules", "url": "https://energy.gov/b", "snippet": "s"}]
    return []  # 'beta' finds nothing -> forces the loop


def fake_fetch(url):
    return PAGE_A if "reuters" in url else PAGE_B


@pytest.fixture()
def result(tmp_path):
    s = Settings("k", "http://x", "m", "m", 2, 2, 4, 6, str(tmp_path / "t.db"))
    llm = FakeLLM()
    log = []
    run_id, report = run_research("Test question?", settings=s, llm=llm, search_fn=fake_search,
                                  fetch_fn=fake_fetch, progress=log.append)
    return s, llm, run_id, report, log


def test_loop_back_happens_and_fills_thin_subquestion(result):
    _, _, _, report, log = result
    assert report["metrics"]["loops"] >= 1
    assert any("looping back" in m for m in log)
    q2 = [s for s in report["sections"] if s["id"] == "Q2"][0]
    assert q2["text"] and not q2["thin"]


def test_unsupported_and_ungrounded_claims_are_rejected_or_marked(result):
    _, _, _, report, _ = result
    rejected = {c["text"]: c["reason"] for c in report["rejected"]}
    unverified = {c["text"]: c["reason"] for c in report["unverified"]}
    assert "Sales tripled by 2030." in rejected
    assert "Totally uncited claim." in rejected and "no retrievable source" in rejected["Totally uncited claim."]
    assert "numbers not found" in unverified["Pack prices fell 45% in 2024."]
    assert "not verbatim" in unverified["Cells are cheap. invented quote"]
    assert {c["text"] for c in report["accepted"]} >= {"Battery pack prices fell 20% in 2024.",
                                                       "New recycling rules raise compliance costs."}


def test_no_uncited_sentence_survives(result):
    _, _, _, report, _ = result
    texts = [report["summary"]] + [s["text"] for s in report["sections"]]
    for t in texts:
        for sent in re.split(r"(?<=\.)\s+(?=[A-Z])", t):
            assert re.search(r"\[\d+\]", sent), sent
    assert "obviously great" not in report["summary"]
    assert report["metrics"]["uncited_sentences_dropped"] >= 3


def test_citations_resolve_to_stored_sources(result):
    _, _, _, report, _ = result
    nums = {b["n"] for b in report["sources"]}
    cited = {int(n) for t in [report["summary"]] + [s["text"] for s in report["sections"]]
             for n in re.findall(r"\[(\d+)\]", t)}
    assert cited and cited <= nums
    assert report["sources"][0]["tier"] in ("primary", "reputable")


def test_metrics_and_contradictions_and_audit_trail(result):
    s, _, run_id, report, _ = result
    m = report["metrics"]
    assert m["claims_total"] == m["claims_verified"] + m["claims_unverified"] + m["claims_rejected"]
    assert 0 < m["groundedness"] < 1
    assert report["contradictions"] and report["contradictions"][0]["claims"]
    from vresearch.store import RunStore
    events = RunStore(s.db_path).load_events(run_id)
    kinds = {(e["step"], e["kind"]) for e in events}
    assert ("verify", "verdict") in kinds and ("coverage", "loop_back") in kinds and ("research", "search") in kinds


def test_followup_reuses_research_and_drops_uncited(result):
    s, llm, run_id, _, _ = result
    out = answer_followup(run_id, "How did prices move?", settings=s, llm=llm)
    assert "Prices fell in 2024" in out["answer"] and "Unsupported aside" not in out["answer"]


def test_exports(result):
    _, _, _, report, _ = result
    assert "Audit trail: rejected claims" in to_markdown(report)
    assert "<h1>" in to_html(report)
    assert to_pdf(report).startswith(b"%PDF")


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    assert _norm("It’s  A Test") == "it's a test"


def test_parallel_research_deterministic_outcomes(tmp_path):
    s = Settings("k", "http://x", "m", "m", 2, 2, 4, 6, str(tmp_path / "parallel.db"), research_workers=3)
    llm = FakeLLM()
    log = []
    run_id, report = run_research("Test question?", settings=s, llm=llm, search_fn=fake_search,
                                  fetch_fn=fake_fetch, progress=log.append)
    assert report["metrics"]["loops"] >= 1
    assert {c["text"] for c in report["accepted"]} >= {"Battery pack prices fell 20% in 2024.",
                                                       "New recycling rules raise compliance costs."}
    rejected = {c["text"] for c in report["rejected"]}
    assert "Sales tripled by 2030." in rejected
    assert "Totally uncited claim." in rejected
    # Check that IDs are unique
    claim_ids = [c["cid"] for c in report["accepted"] + report["unverified"] + report["rejected"]]
    assert len(claim_ids) == len(set(claim_ids))

