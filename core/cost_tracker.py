"""
Cost & Token Tracking — re-export shim around ``self_improving_agent``.

MAP and SIA had nearly-identical cost tracker implementations. The canonical
version now lives in ``self_improving_agent.observability.cost_tracker``;
this module re-exports the same surface so existing MAP callers
(``from core.cost_tracker import attribute, record_usage, ...``) continue to
work unchanged.

On import we register the MAP-specific pricing rows that SIA doesn't ship
(Gemini variants, OpenRouter, Ollama wildcard, OpenAI gpt-4). SIA exposes a
``register_price()`` extension point so this stays a clean addition rather
than a fork.

If/when SIA isn't installed, this module raises a clear ImportError pointing
at the install command — matches the lazy-extra pattern used elsewhere.
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
except ImportError as exc:
    raise ImportError(
        "core.cost_tracker depends on self-improving-agent. Install with:\n"
        "    pip install -e '.[sia]'"
    ) from exc


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
