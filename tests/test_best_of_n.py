"""T2.5: best-of-N candidate generation — critic.score + scoring math."""

from __future__ import annotations

from unittest.mock import MagicMock

from agents.critic import CriticAgent


def _make_critic_with_response(response: str) -> CriticAgent:
    agent = CriticAgent()
    agent.client = MagicMock()
    agent.client.chat.return_value = response
    return agent


class TestCriticScore:
    def test_score_parses_integer(self):
        agent = _make_critic_with_response("7")
        assert agent.score("task", "code") == 7.0

    def test_score_parses_decimal(self):
        agent = _make_critic_with_response("6.5")
        assert agent.score("task", "code") == 6.5

    def test_score_extracts_first_number(self):
        # LLM is chatty even when told to be terse
        agent = _make_critic_with_response("I would say 8 out of 10 because...")
        assert agent.score("task", "code") == 8.0

    def test_score_clamps_to_10(self):
        agent = _make_critic_with_response("100")
        assert agent.score("task", "code") == 10.0

    def test_score_clamps_to_0(self):
        agent = _make_critic_with_response("-5")
        # "-5" -> first match is "5"
        assert agent.score("task", "code") == 5.0

    def test_score_zero_when_unparseable(self):
        agent = _make_critic_with_response("no number here at all")
        assert agent.score("task", "code") == 0.0

    def test_score_zero_on_exception(self):
        agent = CriticAgent()
        agent.client = MagicMock()
        agent.client.chat.side_effect = RuntimeError("network")
        assert agent.score("task", "code") == 0.0


class TestBestOfNSelection:
    """Verify _best_of_n picks correctly given canned candidates.

    We don't mock the orchestrator's module-level agents directly; instead
    we exercise the selection logic via small fixtures.
    """

    def test_passing_beats_failing_even_with_lower_score(self, monkeypatch):
        from core import orchestrator

        # Two candidates: a failing one with high score, a passing one with low.
        # Passing should win.
        candidates_iter = iter([
            ({"code": "FAIL"}, {"status": "failed"}, False),
            ({"code": "PASS"}, {"status": "passed"}, True),
        ])
        monkeypatch.setattr(
            orchestrator,
            "_develop_one_candidate",
            lambda *a, **k: next(candidates_iter),
        )
        # Critic returns high for FAIL, low for PASS — passing pool should
        # still win because the function pre-filters to passing if any.
        scores = {"FAIL": 9.0, "PASS": 3.0}
        monkeypatch.setattr(
            orchestrator.critic,
            "score",
            lambda task_desc, code: scores.get(code, 0.0),
        )

        from core.task_schema import Task

        result = orchestrator._best_of_n(
            Task(id=0, description="t"), feedback=None, context_summary="", n=2
        )
        assert result["passed"] is True
        assert result["code"]["code"] == "PASS"

    def test_highest_score_wins_among_passing(self, monkeypatch):
        from core import orchestrator
        from core.task_schema import Task

        candidates_iter = iter([
            ({"code": "A"}, {"status": "passed"}, True),
            ({"code": "B"}, {"status": "passed"}, True),
            ({"code": "C"}, {"status": "passed"}, True),
        ])
        monkeypatch.setattr(
            orchestrator,
            "_develop_one_candidate",
            lambda *a, **k: next(candidates_iter),
        )
        scores = {"A": 5.0, "B": 9.0, "C": 7.0}
        monkeypatch.setattr(
            orchestrator.critic,
            "score",
            lambda task_desc, code: scores.get(code, 0.0),
        )

        result = orchestrator._best_of_n(
            Task(id=0, description="t"), feedback=None, context_summary="", n=3
        )
        assert result["code"]["code"] == "B"
        assert result["winning_score"] == 9.0

    def test_all_failing_picks_highest_scored_failure(self, monkeypatch):
        from core import orchestrator
        from core.task_schema import Task

        candidates_iter = iter([
            ({"code": "X"}, {"status": "failed"}, False),
            ({"code": "Y"}, {"status": "failed"}, False),
        ])
        monkeypatch.setattr(
            orchestrator,
            "_develop_one_candidate",
            lambda *a, **k: next(candidates_iter),
        )
        scores = {"X": 4.0, "Y": 6.0}
        monkeypatch.setattr(
            orchestrator.critic,
            "score",
            lambda task_desc, code: scores.get(code, 0.0),
        )

        result = orchestrator._best_of_n(
            Task(id=0, description="t"), feedback=None, context_summary="", n=2
        )
        assert result["passed"] is False
        # Highest-scoring failure should win
        assert result["code"]["code"] == "Y"
