"""Core data schemas shared across ProofLens agents and services.

Plain dataclasses are used so the skeleton runs with zero third-party
dependencies. When the pipeline gains real API boundaries these can be
replaced by (or wrapped in) pydantic models without changing call sites.

Every schema provides ``to_dict()``; the top-level session schema also
provides ``from_dict()`` for the JSON storage layer.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


def _now() -> float:
    return time.time()


@dataclass
class Source:
    """A single piece of retrieved material."""

    title: str
    url: str = ""
    snippet: str = ""
    provider: str = "unknown"
    retrieved_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Source":
        return cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            snippet=data.get("snippet", ""),
            provider=data.get("provider", "unknown"),
            retrieved_at=float(data.get("retrieved_at", _now())),
        )


@dataclass
class PlanStep:
    """One action the Researcher should perform."""

    query: str
    intent: str = "search"  # search | fetch | analyze


@dataclass
class Plan:
    """An ordered research plan for a question."""

    question: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    """Material collected for a question."""

    question: str
    sources: list[Source] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Report:
    """A written draft produced from evidence."""

    question: str
    draft: str
    citations: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    """Result of checking a report against its evidence."""

    claim: str
    supported: bool
    score: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchSession:
    """Everything produced while researching one question."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    plan: Optional[Plan] = None
    evidence: Optional[Evidence] = None
    report: Optional[Report] = None
    verdicts: list[Verdict] = field(default_factory=list)
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchSession":
        plan_data = data.get("plan")
        evidence_data = data.get("evidence")
        report_data = data.get("report")
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            question=data.get("question", ""),
            plan=(
                Plan(
                    question=plan_data.get("question", ""),
                    steps=[
                        PlanStep(query=s.get("query", ""), intent=s.get("intent", "search"))
                        for s in plan_data.get("steps", [])
                    ],
                    created_at=float(plan_data.get("created_at", _now())),
                )
                if plan_data
                else None
            ),
            evidence=(
                Evidence(
                    question=evidence_data.get("question", ""),
                    sources=[Source.from_dict(s) for s in evidence_data.get("sources", [])],
                )
                if evidence_data
                else None
            ),
            report=(
                Report(
                    question=report_data.get("question", ""),
                    draft=report_data.get("draft", ""),
                    citations=list(report_data.get("citations", [])),
                    generated_at=float(report_data.get("generated_at", _now())),
                )
                if report_data
                else None
            ),
            verdicts=[
                Verdict(
                    claim=v.get("claim", ""),
                    supported=bool(v.get("supported", False)),
                    score=float(v.get("score", 0.0)),
                    notes=list(v.get("notes", [])),
                )
                for v in data.get("verdicts", [])
            ],
            updated_at=float(data.get("updated_at", _now())),
        )