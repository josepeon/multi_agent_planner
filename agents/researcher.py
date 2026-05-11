"""
Researcher Agent

Runs between Planner and Architect. Takes the planner's task list, derives a
small set of focused queries (library names, framework choices, API specifics
that the architect will need), and returns a research brief that the
Architect injects into its prompt.

When no web search backend is configured (no API key, ``WEB_SEARCH_BACKEND=none``,
or simply running offline) the agent returns an empty brief. The Architect
falls back to its previous behavior cleanly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.cost_tracker import attribute
from core.llm_provider import BaseLLMClient, get_llm_client
from core.web_search import (
    NoOpBackend,
    SearchResult,
    WebSearchBackend,
    get_search_backend,
)


@dataclass
class ResearchBrief:
    """What the researcher hands to the architect."""

    queries: list[str] = field(default_factory=list)
    results: list[SearchResult] = field(default_factory=list)
    summary: str = ""

    @property
    def empty(self) -> bool:
        return not self.queries and not self.results and not self.summary

    def render(self) -> str:
        if self.empty:
            return ""
        lines = ["Research brief from recent web sources:"]
        if self.summary:
            lines.append(self.summary)
        if self.results:
            lines.append("\nSources:")
            for r in self.results[:10]:
                snippet = r.snippet[:200].replace("\n", " ")
                lines.append(f"- {r.title} ({r.url})\n    {snippet}")
        return "\n".join(lines)


class ResearcherAgent:
    """Derives focused queries from a project prompt, searches, summarizes."""

    def __init__(
        self,
        backend: WebSearchBackend | None = None,
        client: BaseLLMClient | None = None,
        max_queries: int = 3,
        results_per_query: int = 3,
    ) -> None:
        self.backend = backend or get_search_backend()
        self.client = client or get_llm_client(temperature=0.2, max_tokens=512)
        self.max_queries = max_queries
        self.results_per_query = results_per_query

    # ----- public -----

    def research(self, prompt: str, tasks: list[str]) -> ResearchBrief:
        """Produce a brief for ``prompt + tasks``. Returns empty brief if
        no backend is configured."""
        if isinstance(self.backend, NoOpBackend) or not self.backend.is_configured():
            return ResearchBrief()

        queries = self._derive_queries(prompt, tasks)
        all_results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for q in queries:
            try:
                with attribute("researcher"):
                    hits = self.backend.search(q, k=self.results_per_query)
            except Exception as exc:
                # Don't fail the pipeline if web search is flaky.
                hits = []
                print(f"  [researcher] search failed for '{q}': {exc}")
            for r in hits:
                if r.url and r.url in seen_urls:
                    continue
                seen_urls.add(r.url)
                all_results.append(r)

        summary = self._summarize(prompt, all_results) if all_results else ""
        return ResearchBrief(queries=queries, results=all_results, summary=summary)

    # ----- internal -----

    def _derive_queries(self, prompt: str, tasks: list[str]) -> list[str]:
        """Ask the LLM for a handful of focused search queries.

        Falls back to a heuristic (the prompt itself) if the LLM call fails,
        so the pipeline keeps moving.
        """
        tasks_block = "\n".join(f"- {t}" for t in tasks[:6])
        system = (
            "You are a research assistant for a code-generation pipeline. "
            "Given a project prompt and its module breakdown, propose a small "
            "number of focused web-search queries that will surface current "
            "best practices, library APIs, and version-specific details the "
            "code generator might otherwise get wrong. Output 2-3 queries, "
            "one per line, no numbering, no quotes."
        )
        user = (
            f"Project: {prompt}\n\n"
            f"Modules:\n{tasks_block}\n\n"
            f"Search queries:"
        )
        try:
            with attribute("researcher"):
                raw = self.client.chat(user_message=user, system_message=system)
        except Exception:
            return [prompt[:200]]

        queries: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            line = re.sub(r"^[\d\.\-•*\s]+", "", line)
            line = line.strip(' "\'')
            if line:
                queries.append(line)
            if len(queries) >= self.max_queries:
                break
        return queries or [prompt[:200]]

    def _summarize(self, prompt: str, results: list[SearchResult]) -> str:
        """Distill the search results into a short brief.

        Uses the LLM but with a tight budget; falls back to a concatenated
        snippet list if the call fails.
        """
        bullets = []
        for r in results[:10]:
            snippet = r.snippet[:300].replace("\n", " ")
            bullets.append(f"- {r.title}: {snippet}")
        bullet_block = "\n".join(bullets)

        system = (
            "Summarize the following web search results for a senior software "
            "architect about to design a Python project. Pull out concrete "
            "facts (library names, API patterns, version notes) and any "
            "common pitfalls. Keep it under 200 words. Do not invent facts "
            "not in the sources."
        )
        user = f"Project: {prompt}\n\nSources:\n{bullet_block}\n\nBrief:"
        try:
            with attribute("researcher"):
                return self.client.chat(user_message=user, system_message=system).strip()
        except Exception:
            return bullet_block
