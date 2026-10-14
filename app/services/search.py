"""Search service with a pluggable provider interface.

The default provider returns no results, keeping the pipeline honest until a
real backend is wired in.

INTEGRATION POINT: implement :class:`SearchProvider` against your search API
(see ``SEARCH_PROVIDER`` / ``SEARCH_API_KEY`` in ``.env.example``) and pass it
to :class:`SearchService`, e.g.::

    service = SearchService(provider=TavilyProvider(api_key=...))
"""
from __future__ import annotations

from typing import Optional

from app.schemas import Source


class SearchProvider:
    """Interface every search provider must implement."""

    name = "base"

    def search(self, query: str) -> list[Source]:
        """Return a list of sources for ``query`` (may be empty)."""
        raise NotImplementedError


class NullProvider(SearchProvider):
    """Placeholder provider: no network access, always returns nothing."""

    name = "none"

    def search(self, query: str) -> list[Source]:
        return []


class SearchService:
    """Facade the Researcher talks to; validates input before provider calls."""

    def __init__(self, provider: Optional[SearchProvider] = None):
        self.provider = provider or NullProvider()

    def search(self, query: str) -> list[Source]:
        text = (query or "").strip()
        if not text:
            return []
        return list(self.provider.search(text))