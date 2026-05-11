"""T2.1: Researcher agent + web search backends."""

from __future__ import annotations

import pytest

from agents.researcher import ResearchBrief, ResearcherAgent
from core.web_search import (
    BraveBackend,
    NoOpBackend,
    SearchResult,
    TavilyBackend,
    get_search_backend,
)


# ===========================================
# Backend factory
# ===========================================

class TestFactory:
    def test_default_with_no_keys_returns_noop(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.delenv("WEB_SEARCH_BACKEND", raising=False)
        backend = get_search_backend()
        assert isinstance(backend, NoOpBackend)

    def test_tavily_key_picks_tavily(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "stub-key")
        monkeypatch.delenv("WEB_SEARCH_BACKEND", raising=False)
        backend = get_search_backend()
        assert isinstance(backend, TavilyBackend)

    def test_brave_key_picks_brave_when_no_tavily(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setenv("BRAVE_API_KEY", "stub-key")
        monkeypatch.delenv("WEB_SEARCH_BACKEND", raising=False)
        backend = get_search_backend()
        assert isinstance(backend, BraveBackend)

    def test_tavily_wins_when_both_set(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "t")
        monkeypatch.setenv("BRAVE_API_KEY", "b")
        monkeypatch.delenv("WEB_SEARCH_BACKEND", raising=False)
        assert isinstance(get_search_backend(), TavilyBackend)

    def test_explicit_none_overrides_keys(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "t")
        monkeypatch.setenv("WEB_SEARCH_BACKEND", "none")
        assert isinstance(get_search_backend(), NoOpBackend)


# ===========================================
# NoOp behavior
# ===========================================

class TestNoOpBackend:
    def test_search_returns_empty(self):
        assert NoOpBackend().search("anything", k=5) == []

    def test_is_configured(self):
        assert NoOpBackend().is_configured() is True


# ===========================================
# Tavily / Brave configuration
# ===========================================

class TestProviderConfiguration:
    def test_tavily_not_configured_without_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        assert TavilyBackend().is_configured() is False

    def test_tavily_configured_with_key(self):
        assert TavilyBackend(api_key="something").is_configured() is True

    def test_brave_search_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="no API key"):
            BraveBackend().search("hi")


# ===========================================
# ResearchBrief
# ===========================================

class TestResearchBrief:
    def test_empty_brief_renders_to_empty_string(self):
        assert ResearchBrief().render() == ""
        assert ResearchBrief().empty is True

    def test_brief_renders_summary_and_sources(self):
        b = ResearchBrief(
            queries=["q1"],
            results=[
                SearchResult(title="Foo", url="http://x", snippet="bar baz"),
            ],
            summary="Stuff worth knowing.",
        )
        out = b.render()
        assert "Stuff worth knowing" in out
        assert "Foo" in out
        assert "http://x" in out


# ===========================================
# ResearcherAgent
# ===========================================

class StubBackend:
    """Test double — returns canned results."""

    name = "stub"

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def is_configured(self) -> bool:
        return True

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return list(self._results)


class StubClient:
    """Test double for the LLM client."""

    def __init__(self, queries_response: str, summary_response: str) -> None:
        self._responses = [queries_response, summary_response]

    def chat(self, *_args, **_kwargs) -> str:
        return self._responses.pop(0) if self._responses else ""


class TestResearcherAgent:
    def test_noop_backend_yields_empty_brief(self):
        agent = ResearcherAgent(backend=NoOpBackend(), client=StubClient("", ""))
        brief = agent.research("anything", ["task1"])
        assert brief.empty

    def test_unconfigured_backend_yields_empty_brief(self):
        # Backend says is_configured=False
        class Unconfigured:
            name = "x"
            def is_configured(self): return False
            def search(self, q, k=5): return []

        agent = ResearcherAgent(backend=Unconfigured(), client=StubClient("", ""))
        assert agent.research("anything", []).empty

    def test_with_results_returns_summary(self):
        backend = StubBackend([
            SearchResult(title="FastAPI Docs", url="https://fastapi.tiangolo.com", snippet="..."),
        ])
        client = StubClient(
            queries_response="fastapi best practices\npydantic v2\n",
            summary_response="Use FastAPI 0.110 with Pydantic v2.",
        )
        agent = ResearcherAgent(backend=backend, client=client)
        brief = agent.research("Build a REST API", ["Define routes"])
        assert not brief.empty
        assert any("fastapi" in q.lower() for q in brief.queries)
        assert "FastAPI" in brief.summary
        assert len(brief.results) >= 1

    def test_dedupes_by_url(self):
        result = SearchResult(title="t", url="https://same.example/x", snippet="a")
        backend = StubBackend([result, result, result])
        client = StubClient("q1\nq2\n", "summary")
        agent = ResearcherAgent(
            backend=backend, client=client, max_queries=2, results_per_query=2
        )
        brief = agent.research("prompt", ["task"])
        # Two queries, but same URL collapsed
        assert len(brief.results) == 1

    def test_search_failure_does_not_crash(self):
        class Boom:
            name = "boom"
            def is_configured(self): return True
            def search(self, *_a, **_kw): raise RuntimeError("network down")

        agent = ResearcherAgent(
            backend=Boom(),
            client=StubClient("q1\n", "summary"),
        )
        brief = agent.research("prompt", ["task"])
        # Should not raise; brief just has no results
        assert brief.results == []
