"""
Human-in-the-loop checkpoints.

The pipeline can pause at well-defined stages (after the Planner emits its
module list, after the Architect emits its design) and ask the user to
approve, edit, or regenerate the artifact before development continues.

Why this matters: most non-trivial failures in the linear pipeline trace
back to a bad architecture decision (wrong storage, wrong framework, wrong
decomposition). Once that decision is locked in, the Developer agent burns
a lot of tokens trying to write code that fits a broken plan. A 10-second
human glance at the plan can save dozens of dev-retry iterations.

This module defines the *protocol* — what the orchestrator hands to the
user and what shape it expects back. Concrete checkpoint handlers (CLI
prompt, web UI form, no-op auto) are plug-in implementations.

Stages:

- ``plan``      — the planner's module list, after the Planner runs
- ``architect`` — the architect's design summary, after the Architect runs

Decisions:

- ``ApproveDecision`` — keep the artifact, continue
- ``EditDecision``    — replace the artifact with the user's version
- ``RegenerateDecision`` — re-run the producing agent with an optional hint

A checkpoint handler is anything with ``async def handle(prompt: CheckpointPrompt) -> CheckpointDecision``
(or its sync equivalent). The default ``AutoApproveHandler`` rubber-stamps
everything, preserving the pre-checkpoint behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# ===========================================
# Protocol types
# ===========================================

CheckpointStage = str  # "plan" | "architect"


@dataclass
class CheckpointPrompt:
    """What the pipeline shows the user at a checkpoint."""

    stage: CheckpointStage
    artifact: Any  # The thing being reviewed (list of tasks, architecture, etc.)
    rendered: str  # Pretty-printed for human display
    context: dict[str, Any] = field(default_factory=dict)  # original prompt etc.


@dataclass
class ApproveDecision:
    pass


@dataclass
class EditDecision:
    """User edited the artifact. ``new_artifact`` replaces what was generated."""

    new_artifact: Any


@dataclass
class RegenerateDecision:
    """User wants the producing agent to try again. Optional hint is
    appended to the agent's prompt on the retry."""

    hint: str | None = None


CheckpointDecision = ApproveDecision | EditDecision | RegenerateDecision


class CheckpointHandler(Protocol):
    """Anything that can answer a CheckpointPrompt."""

    def handle(self, prompt: CheckpointPrompt) -> CheckpointDecision:  # pragma: no cover
        ...


# ===========================================
# Built-in handlers
# ===========================================

class AutoApproveHandler:
    """Default. Approves every checkpoint without prompting.

    Equivalent to the pre-checkpoint behavior; safe to use everywhere.
    """

    def handle(self, prompt: CheckpointPrompt) -> CheckpointDecision:
        return ApproveDecision()


class CLIHandler:
    """Interactive terminal handler.

    Reads from stdin:

    - empty / 'a' / 'approve' / 'y' → approve
    - 'r' / 'regenerate' [hint]     → regenerate with optional hint
    - 'e' / 'edit'                  → opens $EDITOR on the rendered artifact;
                                      the parsed result becomes the new artifact

    The artifact parser is stage-specific; pass a custom one if your stage
    needs special handling.
    """

    def __init__(
        self,
        edit_parser: Callable[[CheckpointStage, str], Any] | None = None,
        input_fn: Callable[[str], str] = input,
        printer: Callable[[str], None] = print,
    ) -> None:
        self._edit_parser = edit_parser or _default_edit_parser
        self._input_fn = input_fn
        self._printer = printer

    def handle(self, prompt: CheckpointPrompt) -> CheckpointDecision:
        self._printer(f"\n{'=' * 60}")
        self._printer(f"CHECKPOINT: {prompt.stage}")
        self._printer(f"{'=' * 60}")
        self._printer(prompt.rendered)
        self._printer(
            "\n[a]pprove  [e]dit  [r]egenerate [hint]  (default: approve)"
        )
        raw = self._input_fn("> ").strip()

        if not raw or raw[0].lower() == "a":
            return ApproveDecision()

        if raw[0].lower() == "r":
            hint = raw[1:].strip() or None
            return RegenerateDecision(hint=hint)

        if raw[0].lower() == "e":
            edited = _edit_in_editor(prompt.rendered)
            try:
                new_artifact = self._edit_parser(prompt.stage, edited)
            except Exception as exc:  # noqa: BLE001 — surface clearly
                self._printer(f"Could not parse edit ({exc}); approving instead.")
                return ApproveDecision()
            return EditDecision(new_artifact=new_artifact)

        # Anything else: approve (don't accidentally block progress).
        return ApproveDecision()


class ScriptedHandler:
    """Deterministic handler for tests and CI.

    Initialized with a sequence of decisions; each ``handle`` call pops the
    next one. Raises if exhausted.
    """

    def __init__(self, decisions: list[CheckpointDecision]) -> None:
        self._decisions = list(decisions)
        self.prompts_seen: list[CheckpointPrompt] = []

    def handle(self, prompt: CheckpointPrompt) -> CheckpointDecision:
        self.prompts_seen.append(prompt)
        if not self._decisions:
            raise RuntimeError("ScriptedHandler ran out of decisions")
        return self._decisions.pop(0)


# ===========================================
# Stage renderers
# ===========================================

def render_plan(tasks: list) -> str:
    """Render a planner output (list of Task or str) for human review."""
    lines = ["Planner proposed these modules:\n"]
    for i, t in enumerate(tasks, 1):
        desc = t.description if hasattr(t, "description") else str(t)
        lines.append(f"  {i}. {desc}")
    return "\n".join(lines)


def render_architecture(architecture: Any) -> str:
    """Render an architecture object. Falls back to str() for unknown types."""
    if hasattr(architecture, "description"):
        return f"Architect proposed:\n\n{architecture.description}"
    return f"Architect proposed:\n\n{architecture}"


def _default_edit_parser(stage: CheckpointStage, edited_text: str) -> Any:
    """Parse the post-edit text back into an artifact.

    Plan stage: each non-empty, non-bullet line becomes a module description.
    Architect stage: the whole text becomes the new architecture description.
    """
    if stage == "plan":
        from core.task_schema import Task

        lines = []
        for raw in edited_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            # Strip leading bullets / numbering
            line = line.lstrip("•-*").strip()
            while line and line[0].isdigit():
                line = line[1:]
            line = line.lstrip(".): ").strip()
            if line:
                lines.append(line)
        return [Task(id=i, description=d) for i, d in enumerate(lines)]

    if stage == "architect":
        # Caller's architecture artifact has a `.description` attribute; we
        # return a thin record that the orchestrator can detect and swap in.
        return _EditedArchitecture(description=edited_text.strip())

    return edited_text.strip()


@dataclass
class _EditedArchitecture:
    description: str

    def get_design_summary(self) -> str:  # parity with real architect output
        return self.description


# ===========================================
# Factory
# ===========================================

def get_handler_from_env() -> CheckpointHandler:
    """Pick a default handler based on env vars.

    - ``CHECKPOINT_MODE=auto`` (default) → AutoApproveHandler
    - ``CHECKPOINT_MODE=cli``           → CLIHandler
    """
    mode = os.environ.get("CHECKPOINT_MODE", "auto").strip().lower()
    if mode == "cli":
        return CLIHandler()
    return AutoApproveHandler()


def _edit_in_editor(initial: str) -> str:
    """Open $EDITOR with initial text; return the saved content. Used by CLIHandler."""
    import subprocess
    import tempfile

    editor = os.environ.get("EDITOR", "vi")
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".txt", delete=False
    ) as f:
        f.write(initial)
        path = f.name
    try:
        subprocess.run([editor, path], check=True)
        with open(path) as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
