"""Verifier agent — checks a report against its evidence.

The default heuristics are deterministic so tests can rely on them:
a report is "supported" when it cites at least half of the known source
URLs and its draft is not empty.

INTEGRATION POINT: add semantic checks (claim extraction, contradiction
detection, source-quality scoring) on top of the citation heuristics.
"""
from __future__ import annotations

from app.schemas import Evidence, Report, Verdict

SUPPORT_THRESHOLD = 0.5


class Verifier:
    """Scores how well a report is backed by its evidence."""

    def verify(self, report: Report, evidence: Evidence) -> Verdict:
        """Return a :class:`Verdict` for ``report`` given ``evidence``."""
        notes: list[str] = []

        known_urls = {s.url for s in evidence.sources if s.url}
        if not evidence.sources:
            notes.append("no evidence sources were collected")
        if not report.draft.strip():
            notes.append("report draft is empty")

        cited = {c for c in report.citations if c}
        covered = cited & known_urls
        missing = known_urls - cited
        if missing:
            notes.append(f"{len(missing)} known source(s) are not cited")

        score = (len(covered) / len(known_urls)) if known_urls else 0.0
        supported = bool(report.draft.strip()) and bool(evidence.sources) and score >= SUPPORT_THRESHOLD

        return Verdict(claim=report.question, supported=supported, score=round(score, 3), notes=notes)