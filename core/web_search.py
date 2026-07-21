"""
Web search backends for the Researcher agent.

The agents pipeline historically generates from training prior alone — APIs
get hallucinated, library versions drift, current best practices are missed.
A Researcher agent that runs a few web searches before the Architect designs
addresses the largest single source of plausible-but-wrong code.

This module defines the ``WebSearchBackend`` protocol and three providers:

  - ``NoOpBackend``   — default. Returns no results. The pipeline runs as
                        before; nothing is required to operate without a key.
  - ``TavilyBackend`` — purpose-built for AI agents. Free tier 1k/mo.
                        Set TAVILY_API_KEY to activate.
  - ``BraveBackend``  — cheaper alternative ($5/mo, 2k searches).
                        Set BRAVE_API_KEY to activate.

Adding more providers later is a one-class change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0  # provider's relevance / freshness score, 0..1
    metadata: dict = field(default_factory=dict)


class WebSearchBackend(Protocol):
    """Minimal protocol — adapters wrap the provider's HTTP API."""

    name: str

    def is_configured(self) -> bool: ...
    def search(self, query: str, k: int = 5) -> list[SearchResult]: ...


# ===========================================
# No-op (default)
# ===========================================


class NoOpBackend:
    """Always returns []. Used when no key is configured.

    Keeps the Researcher agent invocable in any environment; downstream
    consumers see "no results" instead of an import error.
    """

    name = "noop"

    def is_configured(self) -> bool:
        return True

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return []


# ===========================================
# Tavily
# ===========================================


class TavilyBackend:
    """Tavily Search API adapter. Lazy-imports ``tavily``.

    Get a key at https://tavily.com. Set ``TAVILY_API_KEY`` in the env.
    """

    name = "tavily"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _ensure(self):
        if self._client is not None:
            return
        try:
            from tavily import TavilyClient  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "TavilyBackend requires the tavily-python package. "
                "Install with: pip install tavily-python"
            ) from exc
        if not self.api_key:
            raise RuntimeError("TavilyBackend has no API key. Set TAVILY_API_KEY in the env.")
        self._client = TavilyClient(api_key=self.api_key)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self._ensure()
        response = self._client.search(query=query, max_results=k, search_depth="basic")
        results = []
        for r in response.get("results", []):
            results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                )
            )
        return results


# ===========================================
# Brave Search
# ===========================================


class BraveBackend:
    """Brave Search API adapter. Plain HTTP — no SDK dep.

    Get a key at https://api.search.brave.com. Set ``BRAVE_API_KEY`` in the env.
    """

    name = "brave"
    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("BraveBackend has no API key. Set BRAVE_API_KEY in the env.")
        import requests

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {"q": query, "count": k}
        response = requests.get(self._ENDPOINT, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        out = []
        for item in (data.get("web", {}).get("results", []) or [])[:k]:
            out.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                )
            )
        return out


# ===========================================
# Factory
# ===========================================


def get_search_backend() -> WebSearchBackend:
    """Pick a backend based on env. Priority: Tavily > Brave > NoOp.

    Override with ``WEB_SEARCH_BACKEND=tavily|brave|none``.
    """
    explicit = os.environ.get("WEB_SEARCH_BACKEND", "").strip().lower()
    if explicit == "none":
        return NoOpBackend()
    if explicit == "tavily":
        return TavilyBackend()
    if explicit == "brave":
        return BraveBackend()

    # Auto-detect: pick first that has a key.
    if os.environ.get("TAVILY_API_KEY"):
        return TavilyBackend()
    if os.environ.get("BRAVE_API_KEY"):
        return BraveBackend()
    return NoOpBackend()
