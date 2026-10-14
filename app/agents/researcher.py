"""Researcher agent — executes a Plan and collects Evidence.

Uses :class:`app.services.search.SearchService` for retrieval. Wire a real
provider via ``SEARCH_PROVIDER``/``SEARCH_API_KEY`` (see ``.env.example``).

INTEGRATION POINT: pass a :class:`app.services.mesh_client.MeshClient` to
distribute long-running research jobs across the mesh once that service
exists.
"""
from __future__ import annotations

from typing import Optional

from app.schemas import Evidence, Plan, Source
from app.services.search import SearchService


class Researcher:
    """Walks the plan's search steps and aggregates sources."""

    def __init__(self, search_service: Optional[SearchService] = None, mesh_client=None):
        self.search = search_service or SearchService()
        self.mesh_client = mesh_client

    def gather_evidence(self, plan: Plan) -> Evidence:
        """Collect sources for every ``search`` step of the plan."""
        sources: list[Source] = []
        for step in plan.steps:
            if step.intent != "search":
                # "fetch"/"analyze" steps need the real backend; see INTEGRATION POINT.
                continue
            for result in self.search.search(step.query):
                sources.append(
                    Source(
                        title=result.title,
                        url=result.url,
                        snippet=result.snippet,
                        provider=result.provider,
                    )
                )
        return Evidence(question=plan.question, sources=sources)