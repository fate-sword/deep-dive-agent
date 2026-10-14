"""Planner agent — turns a question into an executable research plan.

INTEGRATION POINT: replace the heuristic ``build_plan`` with your planning
strategy of choice (LLM call, prompt template, rule expansion, ...). The
returned ``Plan`` is what ``Researcher`` executes, so keep its shape.
"""
from __future__ import annotations

from app.schemas import Plan, PlanStep


class Planner:
    """Builds a small, ordered research plan for a question."""

    def build_plan(self, question: str) -> Plan:
        """Return a :class:`Plan` with search + analysis steps.

        The default heuristics keep the pipeline deterministic and testable:
        one broad search, one method/context search, one analysis step.
        """
        text = (question or "").strip()
        if not text:
            raise ValueError("question must not be empty")

        steps = [
            PlanStep(query=text, intent="search"),
            PlanStep(query=f"{text} — background and methodology", intent="search"),
            PlanStep(query=text, intent="analyze"),
        ]
        return Plan(question=text, steps=steps)