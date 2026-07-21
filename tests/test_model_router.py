"""T3.1: per-role model routing — defaults, YAML, env overrides."""

from __future__ import annotations

import pytest

from core import model_router as mr
from core.model_router import (
    ModelChoice,
    ModelRouter,
    _parse_model_string,
    _read_yaml_routes,
)


@pytest.fixture(autouse=True)
def _reset_router():
    mr.reset_router()
    yield
    mr.reset_router()


# ===========================================
# Parsing
# ===========================================


class TestParse:
    def test_provider_slash_model(self):
        c = _parse_model_string("groq/llama-3.3-70b-versatile")
        assert c.provider == "groq"
        assert c.model == "llama-3.3-70b-versatile"

    def test_model_only(self):
        c = _parse_model_string("gpt-4o")
        assert c.provider is None
        assert c.model == "gpt-4o"

    def test_whitespace_trimmed(self):
        c = _parse_model_string("  openai / gpt-4o  ")
        assert c.provider == "openai"
        assert c.model == "gpt-4o"


# ===========================================
# YAML loader
# ===========================================


class TestYamlLoader:
    def test_loads_simple_routes(self, tmp_path):
        path = tmp_path / "r.yml"
        path.write_text(
            "# comment\n"
            "planner: groq/llama-3.3-70b-versatile\n"
            "critic: groq/llama-3.1-8b-instant\n"
            "\n"
            "documenter: gpt-4o-mini\n"
        )
        routes = _read_yaml_routes(str(path))
        assert routes["planner"].model == "llama-3.3-70b-versatile"
        assert routes["critic"].model == "llama-3.1-8b-instant"
        assert routes["documenter"].provider is None
        assert routes["documenter"].model == "gpt-4o-mini"

    def test_missing_file_returns_empty(self, tmp_path):
        assert _read_yaml_routes(str(tmp_path / "nope.yml")) == {}


# ===========================================
# ModelRouter
# ===========================================


class TestRouter:
    def test_builtin_defaults_present(self):
        r = ModelRouter()
        assert r.for_role("planner") is not None
        assert r.for_role("documenter") is not None

    def test_unknown_role_returns_none(self):
        r = ModelRouter()
        assert r.for_role("invented") is None

    def test_none_role_returns_none(self):
        r = ModelRouter()
        assert r.for_role(None) is None
        assert r.for_role("") is None

    def test_loaded_routes_override_builtins(self):
        r = ModelRouter(routes={"planner": ModelChoice(provider="openai", model="gpt-4o")})
        choice = r.for_role("planner")
        assert choice.provider == "openai"
        assert choice.model == "gpt-4o"

    def test_env_var_overrides_table(self, monkeypatch):
        r = ModelRouter()
        monkeypatch.setenv("MODEL_FOR_documenter", "openai/gpt-4o-mini")
        choice = r.for_role("documenter")
        assert choice.provider == "openai"
        assert choice.model == "gpt-4o-mini"

    def test_describe_lists_all_routes(self):
        r = ModelRouter()
        out = r.describe()
        assert "planner" in out
        assert "documenter" in out


# ===========================================
# Module-level loader + reset
# ===========================================


class TestModuleLoader:
    def test_get_router_is_cached(self):
        a = mr.get_router()
        b = mr.get_router()
        assert a is b

    def test_reset_invalidates_cache(self):
        a = mr.get_router()
        mr.reset_router()
        b = mr.get_router()
        assert a is not b


# ===========================================
# get_llm_client integration (role -> routing)
# ===========================================


class TestGetLlmClientWithRole:
    def test_role_routes_to_choice_model(self, monkeypatch):
        # We don't construct a real client; just verify config.model gets set
        from core import llm_provider

        captured = {}

        class StubClient(llm_provider.BaseLLMClient):
            def __init__(self, config):
                captured["model"] = config.model
                captured["provider"] = config.provider

            def chat(self, *a, **k):
                return ""

            def chat_with_messages(self, *a, **k):
                return ""

        monkeypatch.setitem(llm_provider.PROVIDERS, "groq", StubClient)
        monkeypatch.setenv("MODEL_FOR_documenter", "groq/llama-3.1-8b-instant")

        llm_provider.get_llm_client(role="documenter")
        assert captured["model"] == "llama-3.1-8b-instant"
        assert captured["provider"] == "groq"

    def test_explicit_model_wins_over_role(self, monkeypatch):
        from core import llm_provider

        captured = {}

        class StubClient(llm_provider.BaseLLMClient):
            def __init__(self, config):
                captured["model"] = config.model

            def chat(self, *a, **k):
                return ""

            def chat_with_messages(self, *a, **k):
                return ""

        monkeypatch.setitem(llm_provider.PROVIDERS, "openai", StubClient)
        # Even though role says one thing, explicit model wins
        monkeypatch.setenv("MODEL_FOR_documenter", "groq/something")
        llm_provider.get_llm_client(role="documenter", provider="openai", model="gpt-4o-mini")
        assert captured["model"] == "gpt-4o-mini"


class TestSchemeParsing:
    def test_scheme_form_parses_provider_correctly(self):
        from core.model_router import _parse_model_string

        choice = _parse_model_string("mlx://path/to/adapter")
        assert choice.provider == "mlx"
        assert choice.model == "path/to/adapter"

    def test_plain_slash_form_still_works(self):
        from core.model_router import _parse_model_string

        choice = _parse_model_string("groq/llama-3.3-70b-versatile")
        assert choice.provider == "groq"
        assert choice.model == "llama-3.3-70b-versatile"

    def test_inline_comment_stripped_from_yaml_value(self, tmp_path):
        from core.model_router import _read_yaml_routes

        cfg = tmp_path / "routes.yml"
        cfg.write_text("planner: groq/llama-3.3-70b-versatile  # the fast one\n")
        routes = _read_yaml_routes(str(cfg))
        assert routes["planner"].model == "llama-3.3-70b-versatile"
