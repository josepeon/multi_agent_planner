"""
Cost & Token Tracking — re-export shim with a bundled fallback.

MAP and SIA had nearly-identical cost tracker implementations. When the
optional ``self-improving-agent`` package is installed (``pip install -e
'.[sia]'``), its canonical ``observability.cost_tracker`` is used. Without
it, the bundled ``core._cost_tracker_local`` implementation (same public
surface) takes over — SIA is an *extra*, and importing core must never
require it. (A hard ImportError here previously broke every base install
and kept CI red.)

On import we register the MAP-specific pricing rows that SIA doesn't ship
(Gemini variants, OpenRouter, Ollama wildcard, OpenAI gpt-4). Both
implementations expose a ``register_price()`` extension point so this stays
a clean addition rather than a fork.
"""

from __future__ import annotations

try:
    from self_improving_agent.observability.cost_tracker import (
        BudgetExceededError,
        CostTracker,
        ModelPrice,
        UsageRecord,
        UsageTotals,
        _current_role,  # used by core/llm_provider.py for role read in record loop
        attribute,
        begin_run,
        get_price,
        get_tracker,
        record_usage,
        register_price,
        render_summary,
        to_dict,
    )

    USING_SIA = True
except ImportError:
    from core._cost_tracker_local import (  # noqa: F401
        BudgetExceededError,
        CostTracker,
        ModelPrice,
        UsageRecord,
        UsageTotals,
        _current_role,
        attribute,
        begin_run,
        get_price,
        get_tracker,
        record_usage,
        register_price,
        render_summary,
        to_dict,
    )

    USING_SIA = False


# ---------------------------------------------------------------------
# Register MAP-specific pricing rows that SIA doesn't ship.
# These rows are model+provider combos MAP supports but SIA doesn't use.
# ---------------------------------------------------------------------

_MAP_EXTRA_PRICES: dict[tuple[str, str], ModelPrice] = {
    # Gemini variants beyond what SIA ships
    ("gemini", "gemini-2.0-flash-exp"): ModelPrice(0.0, 0.0),
    ("gemini", "gemini-1.5-pro"): ModelPrice(1.25, 5.0),
    ("gemini", "gemini-1.5-flash"): ModelPrice(0.075, 0.3),
    # Ollama: local, always free, wildcard match
    ("ollama", "*"): ModelPrice(0.0, 0.0),
    # OpenRouter — pay-per-use; representative llama 3.3 70b instruct
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"): ModelPrice(0.59, 0.79),
    # GPT-4 (SIA only includes 4o and 4o-mini)
    ("openai", "gpt-4"): ModelPrice(30.0, 60.0),
    # Groq fallback models SIA already prices, but list explicitly for clarity
    ("groq", "gemma2-9b-it"): ModelPrice(0.0, 0.0),
    ("groq", "mixtral-8x7b-32768"): ModelPrice(0.0, 0.0),
}

for (provider, model), price in _MAP_EXTRA_PRICES.items():
    register_price(provider, model, price)


__all__ = [
    "BudgetExceededError",
    "CostTracker",
    "ModelPrice",
    "UsageRecord",
    "UsageTotals",
    "_current_role",
    "attribute",
    "begin_run",
    "get_price",
    "get_tracker",
    "record_usage",
    "register_price",
    "render_summary",
    "to_dict",
]
