"""T1.2: Human-in-the-loop checkpoints — handlers, decisions, parsing."""

from __future__ import annotations

import pytest

from core.checkpoints import (
    ApproveDecision,
    AutoApproveHandler,
    CheckpointPrompt,
    CLIHandler,
    EditDecision,
    RegenerateDecision,
    ScriptedHandler,
    _default_edit_parser,
    _EditedArchitecture,
    get_handler_from_env,
    render_architecture,
    render_plan,
)
from core.task_schema import Task

# ===========================================
# AutoApproveHandler
# ===========================================

class TestAutoApprove:
    def test_always_approves(self):
        h = AutoApproveHandler()
        decision = h.handle(
            CheckpointPrompt(stage="plan", artifact=["x"], rendered="rendered")
        )
        assert isinstance(decision, ApproveDecision)


# ===========================================
# ScriptedHandler (for tests)
# ===========================================

class TestScriptedHandler:
    def test_pops_decisions_in_order(self):
        h = ScriptedHandler([
            ApproveDecision(),
            RegenerateDecision(hint="more detail"),
            EditDecision(new_artifact="new"),
        ])
        p1 = CheckpointPrompt(stage="plan", artifact=[], rendered="")
        p2 = CheckpointPrompt(stage="plan", artifact=[], rendered="")
        p3 = CheckpointPrompt(stage="plan", artifact=[], rendered="")

        assert isinstance(h.handle(p1), ApproveDecision)
        assert isinstance(h.handle(p2), RegenerateDecision)
        d3 = h.handle(p3)
        assert isinstance(d3, EditDecision)
        assert d3.new_artifact == "new"

    def test_remembers_prompts(self):
        h = ScriptedHandler([ApproveDecision()])
        p = CheckpointPrompt(stage="architect", artifact={"a": 1}, rendered="r")
        h.handle(p)
        assert h.prompts_seen == [p]

    def test_runs_out(self):
        h = ScriptedHandler([])
        with pytest.raises(RuntimeError, match="ran out"):
            h.handle(CheckpointPrompt(stage="plan", artifact=None, rendered=""))


# ===========================================
# CLIHandler
# ===========================================

class TestCLIHandler:
    def test_empty_input_approves(self):
        out_lines = []
        h = CLIHandler(input_fn=lambda _: "", printer=out_lines.append)
        d = h.handle(CheckpointPrompt(stage="plan", artifact=[], rendered="r"))
        assert isinstance(d, ApproveDecision)

    def test_a_approves(self):
        h = CLIHandler(input_fn=lambda _: "a", printer=lambda _: None)
        d = h.handle(CheckpointPrompt(stage="plan", artifact=[], rendered="r"))
        assert isinstance(d, ApproveDecision)

    def test_r_with_hint(self):
        h = CLIHandler(
            input_fn=lambda _: "r split task 2 into two",
            printer=lambda _: None,
        )
        d = h.handle(CheckpointPrompt(stage="plan", artifact=[], rendered="r"))
        assert isinstance(d, RegenerateDecision)
        assert d.hint == "split task 2 into two"

    def test_r_without_hint(self):
        h = CLIHandler(input_fn=lambda _: "r", printer=lambda _: None)
        d = h.handle(CheckpointPrompt(stage="plan", artifact=[], rendered="r"))
        assert isinstance(d, RegenerateDecision)
        assert d.hint is None

    def test_unknown_input_approves_safely(self):
        h = CLIHandler(input_fn=lambda _: "????", printer=lambda _: None)
        d = h.handle(CheckpointPrompt(stage="plan", artifact=[], rendered="r"))
        assert isinstance(d, ApproveDecision)


# ===========================================
# Renderers
# ===========================================

class TestRenderers:
    def test_render_plan_with_tasks(self):
        tasks = [
            Task(id=0, description="Define data models"),
            Task(id=1, description="Implement core logic"),
        ]
        out = render_plan(tasks)
        assert "1. Define data models" in out
        assert "2. Implement core logic" in out

    def test_render_plan_with_strings(self):
        out = render_plan(["first thing", "second thing"])
        assert "1. first thing" in out
        assert "2. second thing" in out

    def test_render_architecture_with_object(self):
        class Arch:
            description = "Module layout: A, B, C"

        out = render_architecture(Arch())
        assert "Module layout: A, B, C" in out


# ===========================================
# Edit parser
# ===========================================

class TestEditParser:
    def test_plan_parser_splits_lines(self):
        text = """
        1. Define models
        2. Add business logic
        - Build the CLI
        """
        result = _default_edit_parser("plan", text)
        assert len(result) == 3
        assert result[0].description == "Define models"
        assert result[2].description == "Build the CLI"

    def test_plan_parser_skips_blanks(self):
        result = _default_edit_parser("plan", "\n\n\n")
        assert result == []

    def test_architect_parser_returns_wrapper(self):
        result = _default_edit_parser("architect", "  new design here  ")
        assert isinstance(result, _EditedArchitecture)
        assert result.description == "new design here"
        # Parity with real architect output
        assert result.get_design_summary() == "new design here"

    def test_unknown_stage_returns_raw_text(self):
        out = _default_edit_parser("madeup-stage", "  hello  ")
        assert out == "hello"


# ===========================================
# Env-based factory
# ===========================================

class TestEnvFactory:
    def test_default_is_auto(self, monkeypatch):
        monkeypatch.delenv("CHECKPOINT_MODE", raising=False)
        assert isinstance(get_handler_from_env(), AutoApproveHandler)

    def test_cli_mode(self, monkeypatch):
        monkeypatch.setenv("CHECKPOINT_MODE", "cli")
        assert isinstance(get_handler_from_env(), CLIHandler)

    def test_unknown_mode_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setenv("CHECKPOINT_MODE", "madeup")
        assert isinstance(get_handler_from_env(), AutoApproveHandler)
