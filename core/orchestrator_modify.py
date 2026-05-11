"""
Modify-Existing-Codebase Orchestrator

The default pipeline generates a project from scratch. This entry point
modifies an *existing* repo: ingest it, build a RepoMap, run the Architect
with the map injected into its prompt, and produce a list of file changes
(create / modify / delete) rather than a new project tree.

The result is a ``ModificationPlan`` the caller can apply or render as a
diff. We deliberately don't auto-apply: review-then-apply is the safe
default, and ``apply()`` is an explicit, separate call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from core.cost_tracker import attribute, begin_run
from core.repo_ingestion import RepoMap, cleanup_clone, ingest
from core.task_schema import Task


@dataclass
class FileChange:
    """One file-level change in a ModificationPlan."""

    operation: Literal["create", "modify", "delete"]
    path: str
    new_content: str | None = None  # None for delete


@dataclass
class ModificationPlan:
    repo_root: Path
    summary: str
    changes: list[FileChange] = field(default_factory=list)
    raw_response: str = ""

    def render(self) -> str:
        lines = [f"Modification plan for {self.repo_root.name}/"]
        if self.summary:
            lines.append(f"\nSummary: {self.summary}\n")
        for ch in self.changes:
            lines.append(f"  [{ch.operation:>6}] {ch.path}")
        return "\n".join(lines)

    def apply(self, *, dry_run: bool = True) -> dict:
        """Write the planned changes. dry_run=True (default) returns the list
        of paths that *would* be written without touching disk.
        """
        wrote: list[str] = []
        deleted: list[str] = []

        for ch in self.changes:
            target = self.repo_root / ch.path
            if ch.operation == "delete":
                if dry_run:
                    deleted.append(ch.path)
                else:
                    try:
                        target.unlink()
                        deleted.append(ch.path)
                    except FileNotFoundError:
                        pass
                continue

            if ch.new_content is None:
                continue
            if dry_run:
                wrote.append(ch.path)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(ch.new_content)
            wrote.append(ch.path)

        return {"wrote": wrote, "deleted": deleted, "dry_run": dry_run}


# ===========================================
# Plan production
# ===========================================

_CHANGE_BLOCK_RE = re.compile(
    r"^---\s*FILE:\s*(?P<path>[\w./\-_]+)\s*OP:\s*(?P<op>create|modify|delete)\s*---\s*$"
)


def _build_modify_prompt(prompt: str, repo: RepoMap) -> str:
    return (
        f"You are modifying an existing codebase.\n\n"
        f"Repo overview:\n{repo.render_brief()}\n\n"
        f"User request: {prompt}\n\n"
        f"Produce a series of file changes. For each file, emit a header line:\n"
        f"--- FILE: <relative-path> OP: <create|modify|delete> ---\n"
        f"Followed by the FULL new file contents (for create/modify) or "
        f"nothing (for delete).\n"
        f"Separate files with blank lines. Output ONLY these blocks; no other commentary."
    )


def _parse_changes(repo: RepoMap, raw: str) -> list[FileChange]:
    """Parse the LLM's output into FileChange entries.

    The LLM is asked to emit `--- FILE: ... OP: ... ---` headers; we split on
    those, attribute everything between headers to the preceding file.
    """
    changes: list[FileChange] = []
    lines = raw.splitlines()
    i = 0
    current: tuple[str, str] | None = None  # (path, op)
    body: list[str] = []

    def commit():
        if current is None:
            return
        path, op = current
        content: str | None = "\n".join(body).strip()
        if op == "delete":
            content = None
        changes.append(FileChange(operation=op, path=path, new_content=content))

    while i < len(lines):
        match = _CHANGE_BLOCK_RE.match(lines[i].strip())
        if match:
            # Commit previous block
            commit()
            current = (match.group("path"), match.group("op"))
            body = []
        elif current is not None:
            body.append(lines[i])
        i += 1
    commit()
    return changes


def modify_repo(source: str, prompt: str) -> ModificationPlan:
    """Top-level entry: ingest ``source`` and produce a ModificationPlan.

    ``source`` can be a local directory or a git URL. The pipeline does not
    apply changes — the caller decides via ``plan.apply()``.
    """
    repo = ingest(source)
    begin_run()

    # Lazy imports — keeps the symbols un-bound at module top so test patches
    # of core.llm_provider.get_llm_client take effect for fresh instances.
    from agents.architect import ArchitectAgent
    from agents.developer import DeveloperAgent
    from agents.planner import PlannerAgent

    try:
        planner = PlannerAgent()
        with attribute("planner"):
            tasks = planner.plan(prompt)
        task_descs = [t.description if isinstance(t, Task) else str(t) for t in tasks]

        # Architect gets the full repo brief instead of designing from scratch
        architect = ArchitectAgent()
        with attribute("architect"):
            architecture = architect.design(
                prompt,
                task_descs,
                research_brief=f"Repo context:\n{repo.render_brief()}",
            )

        # Developer is invoked once with a single composite "modification" task.
        # We embed the architecture so the change set is coherent across files.
        composite_task = Task(
            id=0,
            description=(
                f"Apply this user request to the existing repo: {prompt}\n\n"
                f"Architecture context:\n{getattr(architecture, 'description', '')}\n\n"
                f"Output format (strict):\n"
                + _build_modify_prompt(prompt, repo).split("Produce", 1)[1]
            ),
        )
        developer = DeveloperAgent()
        with attribute("developer"):
            raw_output = developer.write_code(composite_task.description)

        changes = _parse_changes(repo, raw_output)
        return ModificationPlan(
            repo_root=repo.root,
            summary=getattr(architecture, "description", "")[:300],
            changes=changes,
            raw_response=raw_output,
        )
    except Exception:
        cleanup_clone(repo)
        raise
