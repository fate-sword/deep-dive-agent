"""ProofLens research agents.

Each agent has one clear responsibility:

    Planner     question   -> Plan
    Researcher  Plan       -> Evidence
    Writer      Evidence   -> Report
    Verifier    Report     -> Verdict

`pipeline.run_pipeline` in this package wires them together.
"""
from .planner import Planner
from .researcher import Researcher
from .verifier import Verifier
from .writer import Writer

__all__ = ["Planner", "Researcher", "Verifier", "Writer"]


def run_pipeline(question: str, search_service=None, mesh_client=None) -> dict:
    """Run the full research pipeline for a question and return a session dict.

    This is the integration point for the web layer / API: call it from a
    route or a background job and persist the result with ``app.storage``.
    """
    from app.schemas import ResearchSession

    plan = Planner().build_plan(question)
    evidence = Researcher(search_service=search_service, mesh_client=mesh_client).gather_evidence(plan)
    report = Writer().compose(question, evidence)
    verdict = Verifier().verify(report, evidence)

    session = ResearchSession(question=question, plan=plan, evidence=evidence, report=report, verdicts=[verdict])
    return session.to_dict()