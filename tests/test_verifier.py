"""Tests for the Verifier agent heuristics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.verifier import Verifier
from app.schemas import Evidence, Report, Source


def _evidence(*urls: str) -> Evidence:
    return Evidence(question="q", sources=[Source(title=f"s{i}", url=u) for i, u in enumerate(urls)])


def test_no_evidence_is_not_supported():
    verdict = Verifier().verify(Report(question="q", draft="some text"), Evidence(question="q"))
    assert verdict.supported is False
    assert verdict.score == 0.0
    assert "no evidence sources were collected" in verdict.notes


def test_empty_draft_is_not_supported():
    evidence = _evidence("https://a.example")
    report = Report(question="q", draft="   ", citations=["https://a.example"])
    verdict = Verifier().verify(report, evidence)
    assert verdict.supported is False
    assert "report draft is empty" in verdict.notes


def test_fully_cited_report_is_supported():
    evidence = _evidence("https://a.example", "https://b.example")
    report = Report(question="q", draft="draft", citations=["https://a.example", "https://b.example"])
    verdict = Verifier().verify(report, evidence)
    assert verdict.supported is True
    assert verdict.score == 1.0
    assert not any("not cited" in n for n in verdict.notes)


def test_partial_citations_lower_score_and_note_missing():
    evidence = _evidence("https://a.example", "https://b.example")
    report = Report(question="q", draft="draft", citations=["https://a.example"])
    verdict = Verifier().verify(report, evidence)
    # score sits exactly at the 0.5 threshold, so the report still counts as
    # supported, but the uncited source is flagged:
    assert verdict.score == 0.5
    assert verdict.supported is True
    assert any("not cited" in n for n in verdict.notes)