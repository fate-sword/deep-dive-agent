"""Test that unverified claims NEVER appear in synthesized report text (summary or sections),
while remaining fully preserved in the decision trail, metrics, and export appendices.
"""
import pytest
from vresearch.config import Settings
from vresearch.export import to_html, to_markdown, to_pdf
from vresearch.pipeline import run_research
from vresearch.store import RunStore


class UnverifiedTestFakeLLM:
    calls = 0

    def json(self, system, user, *, model=None, tag="", max_tokens=0):
        self.calls += 1
        if tag == "planner":
            return {
                "sub_questions": [
                    {
                        "id": "Q1",
                        "question": "What are the effects of green tea?",
                        "queries": ["green tea clinical trials"],
                    }
                ]
            }
        if tag == "researcher_refine":
            return {"queries": ["green tea antioxidants"]}
        if tag == "drafter":
            return {
                "claims": [
                    {
                        "text": "Green tea extract reduces oxidative stress in clinical trials.",
                        "evidence_ids": ["E1"],
                    },
                    {
                        "text": "Drinking 10 cups daily prevents 99% of all diseases.",
                        "evidence_ids": ["E1"],
                    },
                ]
            }
        if tag == "relevance":
            return {
                "evaluations": [
                    {"cid": "temp_Q1_1", "relevance": "direct", "reason": "matches inquiry"},
                    {"cid": "temp_Q1_2", "relevance": "direct", "reason": "matches inquiry"},
                ]
            }
        if tag == "verifier":
            # C1 is verified (exact quote in passage)
            if "oxidative stress" in user:
                return {
                    "verdict": "supported",
                    "quote": "Green tea extract reduces oxidative stress in clinical trials.",
                    "reason": "exact verbatim match",
                }
            # C2 is unverified (quote is NOT verbatim in source)
            return {
                "verdict": "supported",
                "quote": "This quote does not exist in source text.",
                "reason": "quote not verbatim in cited source",
            }
        if tag == "synthesizer":
            # Verify that synthesizer input lines only contain verified claims
            assert "99%" not in user, "Unverified claim C2 must NOT be passed to synthesizer!"
            assert "(UNVERIFIED)" not in user, "Tag (UNVERIFIED) must NOT appear in synthesizer prompt!"
            # Simulate a synthesizer that might still attempt to output an unverified sentence
            return {
                "summary": (
                    "Green tea extract reduces oxidative stress in clinical trials [C1]. "
                    "A study on high dosages was flagged as unverified and should be treated with caution [C2] (unverified)."
                ),
                "sections": [
                    {
                        "id": "Q1",
                        "text": (
                            "Clinical trials show oxidative stress reduction [C1]. "
                            "Drinking 10 cups daily prevents 99% of all diseases [C2] (unverified). "
                            "Some findings remain unverified in human trials [C1]."
                        ),
                    }
                ],
            }
        raise AssertionError(f"Unexpected LLM call tag: {tag}")


def test_unverified_claims_never_appear_in_report_text(tmp_path):
    db_file = str(tmp_path / "test_unverified.db")
    s = Settings("k", "http://x", "m", "m", 1, 1, 4, 6, db_path=db_file)
    llm = UnverifiedTestFakeLLM()

    def fake_search(query, n=6):
        return [{"title": "Green Tea Study", "url": "https://pubmed.example.com/1", "snippet": "green tea snippet"}]

    def fake_fetch(url):
        return (
            "Green tea extract reduces oxidative stress in clinical trials. "
            "Volunteers showed improved biomarkers after twelve weeks of supplementation."
        )

    run_id, rep = run_research(
        "What are the effects of green tea?",
        settings=s,
        llm=llm,
        search_fn=fake_search,
        fetch_fn=fake_fetch,
    )

    # 1. Assert unverified content and wording are absent from summary and sections
    summary = rep["summary"]
    assert "Green tea extract reduces oxidative stress" in summary
    assert "99%" not in summary
    assert "prevents" not in summary
    assert "unverified" not in summary.lower()
    assert "(unverified)" not in summary.lower()

    q1_text = rep["sections"][0]["text"]
    assert "Clinical trials show oxidative stress reduction" in q1_text
    assert "99%" not in q1_text
    assert "prevents" not in q1_text
    assert "unverified" not in q1_text.lower()
    assert "(unverified)" not in q1_text.lower()

    # 2. Assert dropped sentences captured the unverified sentences
    dropped = rep["dropped_sentences"]
    assert any("unverified" in d.lower() or "99%" in d for d in dropped)
    assert rep["metrics"]["uncited_sentences_dropped"] >= 2

    # 3. Assert metrics are completely unchanged and accurately record unverified claim
    m = rep["metrics"]
    assert m["claims_total"] == 2
    assert m["claims_verified"] == 1
    assert m["claims_unverified"] == 1
    assert m["claims_rejected"] == 0
    assert m["groundedness"] == 0.5

    # 4. Assert unverified claim is still in report['unverified'] and SQLite audit trail
    assert len(rep["accepted"]) == 1
    assert rep["accepted"][0]["status"] == "verified"
    assert len(rep["unverified"]) == 1
    assert rep["unverified"][0]["status"] == "unverified"
    assert "99%" in rep["unverified"][0]["text"]
    assert "numbers not found" in rep["unverified"][0]["reason"] or "quote not verbatim" in rep["unverified"][0]["reason"]

    trail_claims = RunStore(db_file).load_claims(run_id)
    assert any(c["status"] == "unverified" and "99%" in c["text"] for c in trail_claims)

    # 5. Assert export appendices still list unverified claims, while report body does not
    md = to_markdown(rep)
    # Body must not contain unverified mentions
    md_body = md.split("## Audit trail:")[0]
    assert "unverified" not in md_body.lower()
    # Appendix must list the unverified claim
    md_appendix = md.split("## Audit trail: unverified claims")[1]
    assert "99%" in md_appendix

    html_out = to_html(rep)
    html_body = html_out.split("<h2>Audit:")[0]
    assert "unverified" not in html_body.lower()
    html_appendix = html_out.split("<h2>Audit: Unverified claims</h2>")[1]
    assert "99%" in html_appendix

    pdf_out = to_pdf(rep)
    assert len(pdf_out) > 1000
