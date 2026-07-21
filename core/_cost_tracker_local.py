"""
Cost & Token Tracking — self-contained fallback implementation.

This is the bundled implementation used when the optional
``self-improving-agent`` package isn't installed; ``core.cost_tracker``
prefers SIA's canonical version and falls back to this one. Keep the public
surface identical to SIA's (see core/cost_tracker.py's import list).

Records prompt + completion token counts and estimated USD cost for every
LLM call. Aggregates per-agent and per-run.

Design:

- A single thread-safe ``CostTracker`` instance is shared across the process.
  Agents don't construct it; they call ``record_usage(...)`` directly via the
  helpers in this module.
- Agent attribution is set with a context manager: ``with attribute("planner"):``
  scopes any usage recorded inside to that role. Nested scopes are honored;
  the innermost role wins. Without an explicit scope, calls are attributed
  to ``"unknown"``.
- A run boundary (``begin_run()``) returns a snapshot reset point so the
  orchestrator can report per-run totals separate from process-cumulative.
- Prices are expressed in USD per million tokens. The table covers the
  providers and models the project ships with; unknown models cost $0 and
  are flagged in the summary so we never silently lie about cost.

Token capture lives in ``core.llm_provider`` — each client extracts the
provider's usage payload after a response and calls ``record_usage(...)``.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# ===========================================
# Pricing
# ===========================================

# Prices in USD per 1,000,000 tokens. Sources:
# - Groq: free tier (zeroed; included so the row appears in summaries)
# - OpenAI: gpt-4o public pricing
# - OpenRouter: representative; users can override at runtime
# - Ollama: local, free
# - Gemini: free tier on flash-exp
#
# These figures are deliberately pessimistic when uncertain — better to
# overestimate by a few cents than to under-report.


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_mtok + completion_tokens * self.output_per_mtok
        ) / 1_000_000


_DEFAULT_PRICES: dict[tuple[str, str], ModelPrice] = {
    # Groq: free tier
    ("groq", "llama-3.3-70b-versatile"): ModelPrice(0.0, 0.0),
    ("groq", "llama-3.1-8b-instant"): ModelPrice(0.0, 0.0),
    ("groq", "gemma2-9b-it"): ModelPrice(0.0, 0.0),
    ("groq", "mixtral-8x7b-32768"): ModelPrice(0.0, 0.0),
    # OpenAI
    ("openai", "gpt-4o"): ModelPrice(2.5, 10.0),
    ("openai", "gpt-4o-mini"): ModelPrice(0.15, 0.6),
    ("openai", "gpt-4"): ModelPrice(30.0, 60.0),
    # Gemini (free tier on flash-exp; pro is paid)
    ("gemini", "gemini-2.0-flash-exp"): ModelPrice(0.0, 0.0),
    ("gemini", "gemini-1.5-pro"): ModelPrice(1.25, 5.0),
    ("gemini", "gemini-1.5-flash"): ModelPrice(0.075, 0.3),
    # Ollama local — always free
    ("ollama", "*"): ModelPrice(0.0, 0.0),
    # OpenRouter — pay-per-use; representative llama 3.3 70b instruct
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"): ModelPrice(0.59, 0.79),
}


def get_price(provider: str, model: str) -> ModelPrice | None:
    """Look up a price; falls back to provider-wildcard, else None."""
    if (provider, model) in _DEFAULT_PRICES:
        return _DEFAULT_PRICES[(provider, model)]
    if (provider, "*") in _DEFAULT_PRICES:
        return _DEFAULT_PRICES[(provider, "*")]
    return None


def register_price(provider: str, model: str, price: ModelPrice) -> None:
    """Add or override a pricing row (mirrors SIA's extension point)."""
    _DEFAULT_PRICES[(provider, model)] = price


# ===========================================
# Data model
# ===========================================


@dataclass
class UsageRecord:
    provider: str
    model: str
    agent_role: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    priced: bool  # False if we had no rate card and couldn't price


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    unpriced_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, record: UsageRecord) -> None:
        self.prompt_tokens += record.prompt_tokens
        self.completion_tokens += record.completion_tokens
        self.cost_usd += record.cost_usd
        self.calls += 1
        if not record.priced:
            self.unpriced_calls += 1

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "unpriced_calls": self.unpriced_calls,
        }


# ===========================================
# Tracker
# ===========================================


class CostTracker:
    """Thread-safe accumulator. One instance per process, accessed via the
    module-level helpers below."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[UsageRecord] = []
        self._budget_usd: float | None = None

    # ----- recording -----

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        agent_role: str | None = None,
    ) -> UsageRecord:
        price = get_price(provider, model)
        if price is None:
            cost = 0.0
            priced = False
        else:
            cost = price.cost(prompt_tokens, completion_tokens)
            priced = True

        record = UsageRecord(
            provider=provider,
            model=model,
            agent_role=agent_role or _current_role.get(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            priced=priced,
        )
        with self._lock:
            self._records.append(record)

        if self._budget_usd is not None and self.total().cost_usd > self._budget_usd:
            raise BudgetExceededError(
                f"Run exceeded budget: ${self.total().cost_usd:.4f} > ${self._budget_usd:.4f}"
            )

        return record

    # ----- queries -----

    def all_records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def total(self) -> UsageTotals:
        totals = UsageTotals()
        for r in self.all_records():
            totals.add(r)
        return totals

    def by_agent(self) -> dict[str, UsageTotals]:
        out: dict[str, UsageTotals] = {}
        for r in self.all_records():
            out.setdefault(r.agent_role, UsageTotals()).add(r)
        return out

    def by_model(self) -> dict[str, UsageTotals]:
        out: dict[str, UsageTotals] = {}
        for r in self.all_records():
            key = f"{r.provider}/{r.model}"
            out.setdefault(key, UsageTotals()).add(r)
        return out

    # ----- run lifecycle -----

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._budget_usd = None

    def set_budget(self, usd: float | None) -> None:
        self._budget_usd = usd


class BudgetExceededError(RuntimeError):
    """Raised when an in-progress run crosses its USD budget cap."""


# ===========================================
# Module-level singleton + helpers
# ===========================================

_tracker = CostTracker()
_current_role: contextvars.ContextVar[str] = contextvars.ContextVar("agent_role", default="unknown")


def get_tracker() -> CostTracker:
    return _tracker


def record_usage(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    agent_role: str | None = None,
) -> UsageRecord:
    return _tracker.record(provider, model, prompt_tokens, completion_tokens, agent_role)


@contextmanager
def attribute(role: str) -> Iterator[None]:
    """Scope subsequent record_usage calls to ``role``."""
    token = _current_role.set(role)
    try:
        yield
    finally:
        _current_role.reset(token)


def begin_run(budget_usd: float | None = None) -> None:
    """Reset accumulators at the start of a pipeline run.

    Optionally set a hard budget cap; record_usage will raise BudgetExceededError
    after the threshold is crossed.
    """
    _tracker.reset()
    _tracker.set_budget(budget_usd)


# ===========================================
# Reporting
# ===========================================


def render_summary(tracker: CostTracker | None = None) -> str:
    """Human-readable summary suitable for a terminal footer."""
    tracker = tracker or _tracker
    totals = tracker.total()

    if totals.calls == 0:
        return "Cost: 0 calls (nothing to report)."

    lines = [
        f"Cost: ${totals.cost_usd:.4f}  "
        f"({totals.prompt_tokens:,} prompt + {totals.completion_tokens:,} completion "
        f"= {totals.total_tokens:,} tokens, {totals.calls} call(s))",
    ]

    by_agent = tracker.by_agent()
    if by_agent:
        lines.append("  by agent:")
        for role, t in sorted(by_agent.items(), key=lambda kv: -kv[1].cost_usd):
            lines.append(
                f"    {role:<16} {t.calls:>3} call(s)  {t.total_tokens:>7,} tok  ${t.cost_usd:.4f}"
            )

    by_model = tracker.by_model()
    if len(by_model) > 1:
        lines.append("  by model:")
        for key, t in sorted(by_model.items(), key=lambda kv: -kv[1].cost_usd):
            lines.append(f"    {key:<40} {t.calls:>3} call(s)  ${t.cost_usd:.4f}")

    if totals.unpriced_calls:
        lines.append(
            f"  note: {totals.unpriced_calls} call(s) had no rate card "
            "and contributed $0 to the total."
        )

    return "\n".join(lines)


def to_dict(tracker: CostTracker | None = None) -> dict:
    """Structured snapshot for JSON logs."""
    tracker = tracker or _tracker
    return {
        "total": tracker.total().as_dict(),
        "by_agent": {role: t.as_dict() for role, t in tracker.by_agent().items()},
        "by_model": {key: t.as_dict() for key, t in tracker.by_model().items()},
    }
