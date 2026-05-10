"""T2.7: Cost & token tracking — pricing, attribution, run boundary, budgeting."""

from __future__ import annotations

import threading

import pytest

from core.cost_tracker import (
    BudgetExceededError,
    CostTracker,
    ModelPrice,
    attribute,
    begin_run,
    get_price,
    get_tracker,
    record_usage,
    render_summary,
    to_dict,
)


@pytest.fixture(autouse=True)
def _reset_tracker():
    """Each test starts with a clean shared tracker."""
    begin_run()
    yield
    begin_run()


# ===========================================
# Pricing
# ===========================================

class TestPricing:
    def test_model_price_math(self):
        p = ModelPrice(input_per_mtok=2.5, output_per_mtok=10.0)
        # 1k prompt + 1k completion at 2.5/10.0 per mtok
        assert p.cost(prompt_tokens=1000, completion_tokens=1000) == pytest.approx(
            (2.5 + 10.0) / 1000
        )

    def test_known_model_priced(self):
        price = get_price("openai", "gpt-4o")
        assert price is not None
        assert price.input_per_mtok > 0

    def test_groq_priced_at_zero(self):
        price = get_price("groq", "llama-3.3-70b-versatile")
        assert price is not None
        assert price.input_per_mtok == 0.0

    def test_ollama_wildcard_match(self):
        # Any ollama model should match the "*" entry
        price = get_price("ollama", "totally-made-up-model")
        assert price is not None
        assert price.input_per_mtok == 0.0

    def test_unknown_model_unpriced(self):
        assert get_price("madeup-provider", "madeup-model") is None


# ===========================================
# Recording + attribution
# ===========================================

class TestRecording:
    def test_record_with_default_role(self):
        record_usage("openai", "gpt-4o", 100, 50)
        totals = get_tracker().total()
        assert totals.calls == 1
        assert totals.prompt_tokens == 100
        assert totals.completion_tokens == 50
        assert totals.cost_usd > 0
        # Default attribution
        by_agent = get_tracker().by_agent()
        assert "unknown" in by_agent

    def test_attribute_context(self):
        with attribute("planner"):
            record_usage("groq", "llama-3.3-70b-versatile", 100, 50)
        with attribute("developer"):
            record_usage("groq", "llama-3.3-70b-versatile", 200, 75)

        by_agent = get_tracker().by_agent()
        assert by_agent["planner"].calls == 1
        assert by_agent["developer"].calls == 1
        assert by_agent["planner"].prompt_tokens == 100
        assert by_agent["developer"].prompt_tokens == 200

    def test_nested_attribution_innermost_wins(self):
        with attribute("outer"):
            with attribute("inner"):
                record_usage("groq", "llama-3.3-70b-versatile", 10, 5)

        by_agent = get_tracker().by_agent()
        assert "inner" in by_agent
        assert "outer" not in by_agent

    def test_unpriced_call_marked(self):
        rec = record_usage("madeup-provider", "madeup-model", 100, 50)
        assert rec.priced is False
        assert rec.cost_usd == 0.0
        assert get_tracker().total().unpriced_calls == 1

    def test_by_model_aggregates(self):
        record_usage("openai", "gpt-4o", 100, 100)
        record_usage("openai", "gpt-4o", 200, 100)
        record_usage("groq", "llama-3.3-70b-versatile", 500, 200)

        by_model = get_tracker().by_model()
        assert "openai/gpt-4o" in by_model
        assert "groq/llama-3.3-70b-versatile" in by_model
        assert by_model["openai/gpt-4o"].calls == 2
        assert by_model["openai/gpt-4o"].prompt_tokens == 300


# ===========================================
# Run boundary
# ===========================================

class TestRunBoundary:
    def test_begin_run_resets(self):
        record_usage("openai", "gpt-4o", 100, 100)
        assert get_tracker().total().calls == 1
        begin_run()
        assert get_tracker().total().calls == 0


# ===========================================
# Budget enforcement
# ===========================================

class TestBudget:
    def test_budget_not_exceeded(self):
        begin_run(budget_usd=10.0)
        record_usage("openai", "gpt-4o", 100, 100)  # nowhere near $10
        # No exception

    def test_budget_exceeded_raises(self):
        # gpt-4o = $2.5 in / $10 out per mtok. 1M out = $10.
        begin_run(budget_usd=0.01)
        with pytest.raises(BudgetExceededError):
            # cost = (1000 * 2.5 + 1000 * 10) / 1_000_000 = $0.0125 -> exceeds $0.01
            record_usage("openai", "gpt-4o", 1000, 1000)


# ===========================================
# Reporting
# ===========================================

class TestReporting:
    def test_summary_with_no_calls(self):
        out = render_summary()
        assert "0 calls" in out

    def test_summary_with_calls(self):
        with attribute("planner"):
            record_usage("openai", "gpt-4o", 100, 50)
        with attribute("developer"):
            record_usage("openai", "gpt-4o", 1000, 500)

        out = render_summary()
        assert "Cost:" in out
        assert "planner" in out
        assert "developer" in out
        assert "$" in out

    def test_to_dict_round_trip(self):
        with attribute("critic"):
            record_usage("groq", "llama-3.3-70b-versatile", 50, 25)

        snap = to_dict()
        assert snap["total"]["calls"] == 1
        assert "critic" in snap["by_agent"]
        assert snap["by_agent"]["critic"]["prompt_tokens"] == 50

    def test_summary_flags_unpriced(self):
        record_usage("madeup", "madeup", 100, 100)
        out = render_summary()
        assert "no rate card" in out


# ===========================================
# Thread safety
# ===========================================

class TestThreadSafety:
    def test_concurrent_records(self):
        def worker():
            for _ in range(50):
                record_usage("groq", "llama-3.3-70b-versatile", 10, 5)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        totals = get_tracker().total()
        assert totals.calls == 8 * 50
        assert totals.prompt_tokens == 8 * 50 * 10


# ===========================================
# Isolated tracker instance
# ===========================================

class TestIsolatedInstance:
    """A second CostTracker for tests that don't want to share global state."""

    def test_independent_totals(self):
        local = CostTracker()
        # local doesn't get fed by the module-level record_usage
        record_usage("openai", "gpt-4o", 100, 50)
        assert local.total().calls == 0
        assert get_tracker().total().calls == 1
