"""End-to-end tests for the research pipeline with stub services."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents import Planner, Researcher, Verifier, Writer, run_pipeline
from app.schemas import ResearchSession, Source
from app.services import NullProvider, SearchService
from app.storage import delete_session, load_session, save_session


class StubProvider(NullProvider):
    """Deterministic provider used to exercise the pipeline."""

    name = "stub"

    def search(self, query: str) -> list[Source]:
        return [
            Source(title=f"Result for {query}", url=f"https://example.com/{abs(hash(query)) % 1000}", provider="stub")
        ]


def test_pipeline_runs_end_to_end_with_stub_services():
    session_data = run_pipeline("What is deep research?", search_service=SearchService(StubProvider()))
    session = ResearchSession.from_dict(session_data)

    assert session.question == "What is deep research?"
    assert session.plan is not None and len(session.plan.steps) == 3
    assert session.evidence is not None and len(session.evidence.sources) >= 2
    assert session.report is not None and "Sources reviewed" in session.report.draft
    assert session.verdicts and session.verdicts[0].supported is True


def test_pipeline_with_null_provider_produces_unsupported_report():
    session_data = run_pipeline("Question with no backend", search_service=SearchService())
    session = ResearchSession.from_dict(session_data)

    assert session.evidence is not None and session.evidence.sources == []
    assert session.verdicts and session.verdicts[0].supported is False


def test_storage_roundtrip_and_delete():
    session = ResearchSession(question="storage test")
    session.plan = Planner().build_plan("storage test")

    path = save_session(session)
    try:
        loaded = load_session(session.id)
        assert loaded is not None
        assert loaded.question == "storage test"
        assert loaded.plan is not None and loaded.plan.steps[0].query == "storage test"
    finally:
        assert delete_session(session.id) is True

    assert load_session(session.id) is None