"""
Model Routing

Different agent roles benefit from different models. Planner and Architect
need reasoning; Documenter and parts of the Critic can run on cheap fast
models. Without a router, every agent hits the same default model — paying
70B-class price for tasks a 9B model handles fine.

This module loads a routing table from environment or a YAML config and
hands each agent role a ``(provider, model)`` pair. ``get_llm_client``
gains an optional ``role`` parameter that consults the router.

Config precedence (highest first):

1. Per-role env var: ``MODEL_FOR_<role>`` (e.g. ``MODEL_FOR_documenter``)
   Format: ``provider/model`` or just ``model`` (uses LLM_PROVIDER).
2. ``MODEL_ROUTING`` env var pointing to a YAML file path.
3. ``config/model_routing.yml`` if present.
4. Hardcoded defaults (Groq Llama 70B for everything — same as before).

Without any config, behavior is identical to the pre-router state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    provider: str | None  # None means "honor LLM_PROVIDER env"
    model: str

    def as_kwargs(self) -> dict:
        d = {"model": self.model}
        if self.provider is not None:
            d["provider"] = self.provider
        return d


# Hardcoded defaults. None for provider means "use LLM_PROVIDER env".
_BUILTIN_DEFAULTS: dict[str, ModelChoice] = {
    "planner":      ModelChoice(provider=None, model="llama-3.3-70b-versatile"),
    "architect":    ModelChoice(provider=None, model="llama-3.3-70b-versatile"),
    "developer":    ModelChoice(provider=None, model="llama-3.3-70b-versatile"),
    "critic":       ModelChoice(provider=None, model="llama-3.1-8b-instant"),
    "qa":           ModelChoice(provider=None, model="llama-3.1-8b-instant"),
    "integrator":   ModelChoice(provider=None, model="llama-3.3-70b-versatile"),
    "test_generator": ModelChoice(provider=None, model="llama-3.3-70b-versatile"),
    "documenter":   ModelChoice(provider=None, model="llama-3.1-8b-instant"),
    "researcher":   ModelChoice(provider=None, model="llama-3.1-8b-instant"),
}


def _parse_model_string(raw: str) -> ModelChoice:
    """Parse 'provider://model', 'provider/model', or 'model' into a ModelChoice."""
    raw = raw.strip()
    # Scheme form first: 'mlx://path/to/adapter' must yield provider='mlx',
    # not provider='mlx:' — naive first-'/' splitting broke the documented
    # fine-tuned-adapter routing format.
    if "://" in raw:
        provider, model = raw.split("://", 1)
        return ModelChoice(provider=provider.strip(), model=model.strip())
    if "/" in raw:
        provider, model = raw.split("/", 1)
        return ModelChoice(provider=provider.strip(), model=model.strip())
    return ModelChoice(provider=None, model=raw)


def _read_yaml_routes(path: str) -> dict[str, ModelChoice]:
    """Minimal YAML reader — only flat ``role: model`` mappings.

    We avoid pulling in PyYAML for a config this simple. Format:
        planner: groq/llama-3.3-70b-versatile
        critic: groq/llama-3.1-8b-instant
        # comments allowed
    """
    routes: dict[str, ModelChoice] = {}
    try:
        with open(path) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                # Strip inline comments: 'planner: groq/llama  # fast'
                value = value.split("#", 1)[0].strip()
                if not value:
                    continue
                routes[key.strip().lower()] = _parse_model_string(value)
    except FileNotFoundError:
        pass
    return routes


class ModelRouter:
    """Resolves a role to a (provider, model) pair."""

    def __init__(self, routes: dict[str, ModelChoice] | None = None) -> None:
        self._routes = dict(_BUILTIN_DEFAULTS)
        if routes:
            self._routes.update(routes)

    def for_role(self, role: str | None) -> ModelChoice | None:
        """Return the choice for ``role``, or None if no preference is set.

        Env override ``MODEL_FOR_<role>`` always wins over the loaded table.
        """
        if not role:
            return None
        env_var = f"MODEL_FOR_{role}"
        raw_env = os.environ.get(env_var) or os.environ.get(env_var.upper())
        if raw_env:
            return _parse_model_string(raw_env)
        return self._routes.get(role.lower())

    def describe(self) -> str:
        lines = ["Model routing:"]
        for role, choice in sorted(self._routes.items()):
            provider = choice.provider or "<default>"
            lines.append(f"  {role:<16} -> {provider}/{choice.model}")
        return "\n".join(lines)


# ===========================================
# Module-level loader
# ===========================================

_router: ModelRouter | None = None


def get_router() -> ModelRouter:
    """Return the process-wide router, building it on first call."""
    global _router
    if _router is not None:
        return _router

    routes: dict[str, ModelChoice] = {}

    # Lowest priority: file at default path
    default_path = os.path.join("config", "model_routing.yml")
    if os.path.exists(default_path):
        routes.update(_read_yaml_routes(default_path))

    # Override: explicit file via env
    env_path = os.environ.get("MODEL_ROUTING")
    if env_path and os.path.exists(env_path):
        routes.update(_read_yaml_routes(env_path))

    _router = ModelRouter(routes)
    return _router


def reset_router() -> None:
    """Drop the cached router so tests / dynamic reloads pick up env changes."""
    global _router
    _router = None
