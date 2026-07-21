"""
Deployer Agent

Closes the loop: after the pipeline generates code + tests + README, the
Deployer figures out what kind of project this is, generates the appropriate
deployment artifacts (Procfile / vercel.json / modal_app.py / etc.), and —
optionally — runs the deploy command to bring it live.

Design:

- Each deploy target is a ``DeployTarget`` adapter (Railway, Modal, Vercel,
  Streamlit Cloud, PyPI). Targets declare their applicability via
  ``matches(project)`` so the Deployer can auto-pick.
- ``DeployerAgent.plan(project_dir)`` inspects the generated project and
  returns a ``DeploymentPlan``: which target, what files to write, what
  command(s) to run.
- ``DeployerAgent.execute(plan, dry_run=False)`` writes the files. By default
  it does NOT run network commands — those require explicit ``dry_run=False``
  AND a configured backend (e.g. a Railway token in the env). This is
  deliberate: we never deploy without the user opting in.

Without any deploy tokens configured, the deployer still emits the artifact
files (Procfile, vercel.json, etc.) so the project is deploy-ready by hand.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# ===========================================
# Project introspection
# ===========================================

@dataclass
class ProjectShape:
    """What we can tell about the generated project from its files."""

    root: Path
    files: list[str] = field(default_factory=list)
    has_flask: bool = False
    has_fastapi: bool = False
    has_streamlit: bool = False
    has_click: bool = False
    has_typer: bool = False
    has_main: bool = False
    has_pyproject: bool = False
    has_requirements: bool = False

    @property
    def kind(self) -> str:
        """High-level project kind, used for human-readable summaries."""
        if self.has_streamlit:
            return "streamlit_app"
        if self.has_flask or self.has_fastapi:
            return "web_service"
        if self.has_click or self.has_typer:
            return "cli"
        return "library"


_FLASK_RE = re.compile(r"\bfrom flask\b|\bimport flask\b", re.I)
_FASTAPI_RE = re.compile(r"\bfrom fastapi\b|\bimport fastapi\b", re.I)
_STREAMLIT_RE = re.compile(r"\bimport streamlit\b|\bstreamlit as st\b", re.I)
_CLICK_RE = re.compile(r"\bimport click\b|\bfrom click\b", re.I)
_TYPER_RE = re.compile(r"\bimport typer\b|\bfrom typer\b", re.I)
_MAIN_RE = re.compile(r"if __name__\s*==\s*['\"]__main__['\"]")


def inspect_project(root: str | Path) -> ProjectShape:
    """Walk ``root`` looking for telltale imports and entry points."""
    root_path = Path(root)
    if not root_path.exists():
        return ProjectShape(root=root_path)

    shape = ProjectShape(root=root_path)
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root_path).as_posix()
        shape.files.append(rel)
        if path.name == "pyproject.toml":
            shape.has_pyproject = True
        if path.name == "requirements.txt":
            shape.has_requirements = True
        if path.suffix != ".py":
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if _FLASK_RE.search(text):
            shape.has_flask = True
        if _FASTAPI_RE.search(text):
            shape.has_fastapi = True
        if _STREAMLIT_RE.search(text):
            shape.has_streamlit = True
        if _CLICK_RE.search(text):
            shape.has_click = True
        if _TYPER_RE.search(text):
            shape.has_typer = True
        if _MAIN_RE.search(text):
            shape.has_main = True
    return shape


# ===========================================
# Deploy plan
# ===========================================

@dataclass
class DeploymentArtifact:
    """A file the deployer will write before deploying."""

    relative_path: str
    content: str


@dataclass
class DeploymentPlan:
    target: str  # e.g. "railway", "modal", "vercel", "streamlit", "pypi", "manual"
    project_kind: str
    artifacts: list[DeploymentArtifact] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    notes: str = ""

    def summary(self) -> str:
        lines = [f"Target: {self.target} ({self.project_kind})"]
        if self.artifacts:
            lines.append("Files to write:")
            for a in self.artifacts:
                lines.append(f"  - {a.relative_path}")
        if self.commands:
            lines.append("Commands:")
            for cmd in self.commands:
                lines.append("  - " + " ".join(cmd))
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


# ===========================================
# Targets
# ===========================================

class DeployTarget(Protocol):
    name: str

    def matches(self, project: ProjectShape) -> bool: ...
    def plan(self, project: ProjectShape) -> DeploymentPlan: ...


class RailwayTarget:
    """Railway: any Flask/FastAPI web service. Emits Procfile + railway.toml."""

    name = "railway"

    def matches(self, project: ProjectShape) -> bool:
        return project.has_flask or project.has_fastapi

    def plan(self, project: ProjectShape) -> DeploymentPlan:
        # Pick a sensible default startup command
        if project.has_fastapi:
            start_cmd = (
                "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
            )
        else:
            start_cmd = "gunicorn -b 0.0.0.0:${PORT:-8000} main:app"

        artifacts = [
            DeploymentArtifact(
                relative_path="Procfile",
                content=f"web: {start_cmd}\n",
            ),
            DeploymentArtifact(
                relative_path="railway.toml",
                content=(
                    "[deploy]\n"
                    f"startCommand = \"{start_cmd}\"\n"
                    "healthcheckPath = \"/\"\n"
                    "restartPolicyType = \"on_failure\"\n"
                    "restartPolicyMaxRetries = 3\n"
                ),
            ),
        ]

        # Only run the deploy command if a token is configured
        token = os.environ.get("RAILWAY_TOKEN")
        commands: list[list[str]] = []
        notes = ""
        if token:
            commands = [["railway", "up"]]
            notes = "RAILWAY_TOKEN detected — will run `railway up` on execute."
        else:
            notes = (
                "RAILWAY_TOKEN not set; files written but deploy skipped. "
                "Install the railway CLI and `railway link` your project to "
                "complete the deploy by hand."
            )

        return DeploymentPlan(
            target=self.name,
            project_kind=project.kind,
            artifacts=artifacts,
            commands=commands,
            notes=notes,
        )


class StreamlitCloudTarget:
    """Streamlit Cloud: deploys via GitHub. Emits requirements + config."""

    name = "streamlit_cloud"

    def matches(self, project: ProjectShape) -> bool:
        return project.has_streamlit

    def plan(self, project: ProjectShape) -> DeploymentPlan:
        artifacts = []
        if not project.has_requirements:
            artifacts.append(
                DeploymentArtifact(
                    relative_path="requirements.txt",
                    content="streamlit>=1.30\n",
                )
            )
        artifacts.append(
            DeploymentArtifact(
                relative_path=".streamlit/config.toml",
                content="[server]\nheadless = true\n",
            )
        )
        return DeploymentPlan(
            target=self.name,
            project_kind=project.kind,
            artifacts=artifacts,
            commands=[],
            notes=(
                "Streamlit Cloud deploys from a public GitHub repo. After "
                "writing these files, push to GitHub and create the app at "
                "https://share.streamlit.io."
            ),
        )


class VercelTarget:
    """Vercel: catches frontend / static / function projects.

    Default impl handles Python serverless functions in /api. Falls through
    if no api/ directory exists.
    """

    name = "vercel"

    def matches(self, project: ProjectShape) -> bool:
        return any(f.startswith("api/") for f in project.files)

    def plan(self, project: ProjectShape) -> DeploymentPlan:
        return DeploymentPlan(
            target=self.name,
            project_kind=project.kind,
            artifacts=[
                DeploymentArtifact(
                    relative_path="vercel.json",
                    content=(
                        '{\n'
                        '  "version": 2,\n'
                        '  "builds": [\n'
                        '    {"src": "api/*.py", "use": "@vercel/python"}\n'
                        '  ]\n'
                        '}\n'
                    ),
                )
            ],
            commands=[],
            notes="Run `vercel` from the project root once linked.",
        )


class ModalTarget:
    """Modal: works for Python services and long-running jobs.

    Generates a minimal modal_app.py that wraps the user's main.py.
    """

    name = "modal"

    def matches(self, project: ProjectShape) -> bool:
        # We never auto-match Modal as preferred; treated as opt-in via
        # DEPLOY_TARGET=modal. matches() returns False so the auto-picker
        # falls through to manual.
        return False

    def plan(self, project: ProjectShape) -> DeploymentPlan:
        artifact = DeploymentArtifact(
            relative_path="modal_app.py",
            content=(
                "import modal\n\n"
                "image = modal.Image.debian_slim().pip_install_from_requirements(\n"
                "    \"requirements.txt\" if __import__('os').path.exists(\"requirements.txt\") else []\n"
                ")\n\n"
                "app = modal.App(\"generated-project\", image=image)\n\n"
                "@app.function()\n"
                "def main():\n"
                "    from main import main as user_main\n"
                "    return user_main()\n"
            ),
        )
        return DeploymentPlan(
            target=self.name,
            project_kind=project.kind,
            artifacts=[artifact],
            commands=[["modal", "deploy", "modal_app.py"]],
            notes="Requires `pip install modal` and `modal token new`.",
        )


class ManualTarget:
    """Default fallback: nothing automated, just a hint."""

    name = "manual"

    def matches(self, project: ProjectShape) -> bool:
        return True

    def plan(self, project: ProjectShape) -> DeploymentPlan:
        return DeploymentPlan(
            target=self.name,
            project_kind=project.kind,
            artifacts=[],
            commands=[],
            notes=(
                f"No automatic deploy target matched for a {project.kind} project. "
                f"Run with DEPLOY_TARGET=railway|streamlit_cloud|vercel|modal to force one."
            ),
        )


_TARGETS: dict[str, DeployTarget] = {
    "railway": RailwayTarget(),
    "streamlit_cloud": StreamlitCloudTarget(),
    "vercel": VercelTarget(),
    "modal": ModalTarget(),
    "manual": ManualTarget(),
}


# ===========================================
# Agent
# ===========================================

class DeployerAgent:
    """Decides the target, writes deployment artifacts, optionally deploys."""

    def __init__(self, targets: list[DeployTarget] | None = None) -> None:
        self._targets = targets or [
            RailwayTarget(),
            StreamlitCloudTarget(),
            VercelTarget(),
            # ModalTarget intentionally not auto-matched
            ManualTarget(),
        ]

    def plan(self, project_dir: str | Path) -> DeploymentPlan:
        """Inspect ``project_dir`` and choose a deploy target.

        Honors DEPLOY_TARGET=<name> to override auto-detection.
        """
        project = inspect_project(project_dir)
        forced = os.environ.get("DEPLOY_TARGET", "").strip().lower()
        if forced:
            target = _TARGETS.get(forced) or ManualTarget()
            return target.plan(project)

        for target in self._targets:
            if target.matches(project):
                return target.plan(project)
        return ManualTarget().plan(project)

    def execute(
        self,
        plan: DeploymentPlan,
        project_dir: str | Path,
        *,
        dry_run: bool = True,
    ) -> dict:
        """Write the plan's artifacts; optionally run the deploy commands.

        Returns ``{"wrote": [paths], "ran": [cmds], "skipped": [cmds]}``.

        ``dry_run=True`` (default) writes files but does NOT run network
        commands. The deployer never touches the user's account without an
        explicit ``dry_run=False`` AND a configured token.
        """
        project_root = Path(project_dir)
        project_root.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        ran: list[list[str]] = []
        skipped: list[list[str]] = []

        for artifact in plan.artifacts:
            target_path = project_root / artifact.relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(artifact.content)
            written.append(str(target_path.relative_to(project_root)))

        if dry_run:
            skipped = list(plan.commands)
        else:
            import subprocess

            for cmd in plan.commands:
                try:
                    subprocess.run(cmd, cwd=project_root, check=True)
                    ran.append(cmd)
                except Exception as exc:  # noqa: BLE001
                    skipped.append(cmd + [f"# failed: {exc}"])

        return {
            "wrote": written,
            "ran": ran,
            "skipped": skipped,
            "target": plan.target,
        }
