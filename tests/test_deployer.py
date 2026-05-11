"""T2.4: DeployerAgent — project introspection, plan selection, artifact writing."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.deployer import (
    DeployerAgent,
    ManualTarget,
    RailwayTarget,
    StreamlitCloudTarget,
    VercelTarget,
    inspect_project,
)


# ===========================================
# Project introspection
# ===========================================

class TestInspection:
    def test_empty_dir(self, tmp_path):
        shape = inspect_project(tmp_path)
        assert shape.kind == "library"
        assert not (shape.has_flask or shape.has_fastapi or shape.has_streamlit)

    def test_detects_flask(self, tmp_path):
        (tmp_path / "main.py").write_text("from flask import Flask\napp = Flask(__name__)\n")
        shape = inspect_project(tmp_path)
        assert shape.has_flask
        assert shape.kind == "web_service"

    def test_detects_fastapi(self, tmp_path):
        (tmp_path / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
        shape = inspect_project(tmp_path)
        assert shape.has_fastapi
        assert shape.kind == "web_service"

    def test_detects_streamlit(self, tmp_path):
        (tmp_path / "app.py").write_text("import streamlit as st\nst.title('Hi')\n")
        shape = inspect_project(tmp_path)
        assert shape.has_streamlit
        assert shape.kind == "streamlit_app"

    def test_detects_cli(self, tmp_path):
        (tmp_path / "main.py").write_text("import click\n@click.command()\ndef cli():\n    pass\n")
        shape = inspect_project(tmp_path)
        assert shape.has_click
        assert shape.kind == "cli"

    def test_detects_requirements_and_pyproject(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        shape = inspect_project(tmp_path)
        assert shape.has_requirements
        assert shape.has_pyproject

    def test_nonexistent_dir_returns_empty_shape(self, tmp_path):
        shape = inspect_project(tmp_path / "does-not-exist")
        assert shape.kind == "library"


# ===========================================
# Targets
# ===========================================

class TestRailwayTarget:
    def test_matches_flask(self, tmp_path):
        (tmp_path / "main.py").write_text("from flask import Flask\napp = Flask(__name__)")
        shape = inspect_project(tmp_path)
        assert RailwayTarget().matches(shape) is True

    def test_does_not_match_pure_cli(self, tmp_path):
        (tmp_path / "main.py").write_text("import click\n")
        shape = inspect_project(tmp_path)
        assert RailwayTarget().matches(shape) is False

    def test_plan_includes_procfile(self, tmp_path):
        (tmp_path / "main.py").write_text("from fastapi import FastAPI")
        shape = inspect_project(tmp_path)
        plan = RailwayTarget().plan(shape)
        files = {a.relative_path for a in plan.artifacts}
        assert "Procfile" in files
        assert "railway.toml" in files

    def test_uses_uvicorn_for_fastapi(self, tmp_path):
        (tmp_path / "main.py").write_text("from fastapi import FastAPI")
        plan = RailwayTarget().plan(inspect_project(tmp_path))
        procfile = next(a for a in plan.artifacts if a.relative_path == "Procfile")
        assert "uvicorn" in procfile.content

    def test_uses_gunicorn_for_flask(self, tmp_path):
        (tmp_path / "main.py").write_text("from flask import Flask")
        plan = RailwayTarget().plan(inspect_project(tmp_path))
        procfile = next(a for a in plan.artifacts if a.relative_path == "Procfile")
        assert "gunicorn" in procfile.content

    def test_no_command_without_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("RAILWAY_TOKEN", raising=False)
        (tmp_path / "main.py").write_text("from flask import Flask")
        plan = RailwayTarget().plan(inspect_project(tmp_path))
        assert plan.commands == []
        assert "RAILWAY_TOKEN" in plan.notes

    def test_command_present_with_token(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAILWAY_TOKEN", "stub")
        (tmp_path / "main.py").write_text("from flask import Flask")
        plan = RailwayTarget().plan(inspect_project(tmp_path))
        assert plan.commands == [["railway", "up"]]


class TestStreamlitTarget:
    def test_matches_streamlit(self, tmp_path):
        (tmp_path / "app.py").write_text("import streamlit as st")
        assert StreamlitCloudTarget().matches(inspect_project(tmp_path))

    def test_emits_requirements_when_missing(self, tmp_path):
        (tmp_path / "app.py").write_text("import streamlit as st")
        plan = StreamlitCloudTarget().plan(inspect_project(tmp_path))
        files = {a.relative_path for a in plan.artifacts}
        assert "requirements.txt" in files

    def test_skips_requirements_if_present(self, tmp_path):
        (tmp_path / "app.py").write_text("import streamlit as st")
        (tmp_path / "requirements.txt").write_text("streamlit\n")
        plan = StreamlitCloudTarget().plan(inspect_project(tmp_path))
        files = {a.relative_path for a in plan.artifacts}
        assert "requirements.txt" not in files


# ===========================================
# Agent end-to-end
# ===========================================

class TestDeployerAgent:
    def test_plan_picks_railway_for_flask(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEPLOY_TARGET", raising=False)
        (tmp_path / "main.py").write_text("from flask import Flask")
        agent = DeployerAgent()
        plan = agent.plan(tmp_path)
        assert plan.target == "railway"

    def test_plan_picks_streamlit_for_streamlit(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEPLOY_TARGET", raising=False)
        (tmp_path / "app.py").write_text("import streamlit as st")
        plan = DeployerAgent().plan(tmp_path)
        assert plan.target == "streamlit_cloud"

    def test_plan_falls_back_to_manual(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEPLOY_TARGET", raising=False)
        (tmp_path / "lib.py").write_text("def add(a, b): return a + b\n")
        plan = DeployerAgent().plan(tmp_path)
        assert plan.target == "manual"

    def test_deploy_target_env_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEPLOY_TARGET", "vercel")
        plan = DeployerAgent().plan(tmp_path)
        assert plan.target == "vercel"

    def test_execute_writes_artifacts_dry_run(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RAILWAY_TOKEN", raising=False)
        (tmp_path / "main.py").write_text("from flask import Flask")
        agent = DeployerAgent()
        plan = agent.plan(tmp_path)
        result = agent.execute(plan, tmp_path, dry_run=True)

        assert "Procfile" in result["wrote"]
        assert (tmp_path / "Procfile").exists()
        # Dry-run should skip commands, but in this case there are none anyway
        assert result["ran"] == []

    def test_execute_dry_run_does_not_execute_commands(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RAILWAY_TOKEN", "stub")
        (tmp_path / "main.py").write_text("from flask import Flask")
        agent = DeployerAgent()
        plan = agent.plan(tmp_path)
        result = agent.execute(plan, tmp_path, dry_run=True)

        assert result["ran"] == []
        # The command should be in the skipped list
        assert ["railway", "up"] in result["skipped"]
