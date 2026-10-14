"""Writer agent — composes Evidence into a draft report.

INTEGRATION POINT: replace the template composer with your model-backed
writer. Keep returning a :class:`app.schemas.Report` so the Verifier and
storage layer keep working unchanged.
"""
from __future__ import annotations

from app.schemas import Evidence, Report


class Writer:
    """Renders evidence into a plain-text research brief."""

    def compose(self, question: str, evidence: Evidence) -> Report:
        """Build a deterministic draft from the collected sources."""
        lines = [f"Research brief: {(question or evidence.question).strip()}", ""]
        sources = list(evidence.sources)
        if not sources:
            lines.append("No sources were collected for this question yet.")
            lines.append("Wire a search provider (see app/services/search.py) to enable retrieval.")
        else:
            lines.append(f"Sources reviewed: {len(sources)}")
            lines.append("")
            for index, source in enumerate(sources[:10], start=1):
                label = source.title or "(untitled source)"
                location = source.url or source.snippet
                lines.append(f"[{index}] {label}" + (f" — {location}" if location else ""))

        draft = "\n".join(lines)
        citations = [s.url for s in sources if s.url]
        return Report(question=question, draft=draft, citations=citations)