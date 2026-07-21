"""
Repo Ingestion

For modify-existing-codebase mode. Takes a directory or git URL and produces
a ``RepoMap`` — a structured summary of the project the pipeline is about to
modify. The Architect uses this map to understand what already exists before
proposing changes; the Developer references it when writing modifications.

What we extract:

- file inventory (paths + size)
- per-file AST summary: classes (with methods), top-level functions, imports
- README / pyproject / requirements snippets

What we explicitly DON'T do here:

- run the project's tests
- execute any of its code
- mutate the repo (read-only)

Cloning a git URL uses ``git`` directly via subprocess — no extra dep on
GitPython, and shallow clone keeps it fast.
"""

from __future__ import annotations

import ast
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Files we never read in full (binaries, build artifacts, etc.)
_BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".mp3",
    ".mp4",
    ".webm",
    ".mov",
    ".wav",
}

_SKIP_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    ".mypy_cache",
    "coverage",
    "htmlcov",
    ".cache",
}


@dataclass
class FileSummary:
    path: str
    size_bytes: int
    classes: list[str] = field(default_factory=list)  # "ClassName(methods=[...])"
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


@dataclass
class RepoMap:
    """Read-only map of a codebase."""

    root: Path
    files: list[FileSummary] = field(default_factory=list)
    readme: str = ""
    pyproject: str = ""
    requirements: list[str] = field(default_factory=list)

    def file_count(self) -> int:
        return len(self.files)

    def python_files(self) -> list[FileSummary]:
        return [f for f in self.files if f.path.endswith(".py")]

    def render_brief(self, max_files: int = 40) -> str:
        """Compact representation suitable for injection into an LLM prompt."""
        lines = [f"Repo at {self.root.name}/ — {self.file_count()} file(s)"]

        if self.readme:
            head = self.readme.splitlines()[:8]
            lines.append("README excerpt:")
            for h in head:
                lines.append(f"  {h}")

        if self.requirements:
            lines.append("requirements.txt:")
            for r in self.requirements[:12]:
                lines.append(f"  - {r}")

        py = self.python_files()
        if py:
            lines.append("Python modules:")
            for f in py[:max_files]:
                detail = []
                if f.classes:
                    detail.append("classes: " + ", ".join(f.classes[:5]))
                if f.functions:
                    detail.append("funcs: " + ", ".join(f.functions[:8]))
                tail = "; ".join(detail) if detail else f"{f.size_bytes}B"
                lines.append(f"  - {f.path}  ({tail})")
            if len(py) > max_files:
                lines.append(f"  ... and {len(py) - max_files} more")

        return "\n".join(lines)


# ===========================================
# Ingestion
# ===========================================


def ingest(source: str) -> RepoMap:
    """Build a RepoMap from a local path or a git URL.

    Git URLs are shallow-cloned to a temp directory; that directory is the
    RepoMap's root. Callers that need to clean up should call
    ``cleanup_clone(repo_map)`` when done.
    """
    if _looks_like_git_url(source):
        _validate_git_url(source)
        clone_dir = tempfile.mkdtemp(prefix="map_repo_clone_")
        subprocess.run(
            # "--" stops git from parsing a hostile source as an option
            ["git", "clone", "--depth", "1", "--", source, clone_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            # Block ext::/file:// and other command-executing transports
            env={**os.environ, "GIT_ALLOW_PROTOCOL": "https:http:ssh:git"},
        )
        return _ingest_dir(Path(clone_dir))

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if not path.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {source}")
    return _ingest_dir(path)


def cleanup_clone(repo_map: RepoMap) -> None:
    """If the RepoMap came from a temp clone, remove it. Safe no-op otherwise."""
    if str(repo_map.root).startswith(tempfile.gettempdir()):
        import shutil

        shutil.rmtree(repo_map.root, ignore_errors=True)


def _looks_like_git_url(source: str) -> bool:
    return source.startswith(
        ("git@", "https://github.com", "http://github.com")
    ) or source.endswith(".git")


_ALLOWED_URL_PREFIXES = ("https://", "http://", "git@", "ssh://", "git://")


def _validate_git_url(source: str) -> None:
    """Reject sources git would treat as options or command-executing transports.

    ``ext::sh -c '...'`` is a valid git "URL" that runs the embedded command,
    and a leading ``-`` can be parsed as a git option — both are RCE vectors
    when the source string comes from an untrusted request.
    """
    if source.startswith("-"):
        raise ValueError(f"Refusing git source that looks like an option: {source!r}")
    if not source.startswith(_ALLOWED_URL_PREFIXES):
        raise ValueError(
            f"Refusing git source with unsupported transport: {source!r} "
            f"(allowed: {', '.join(_ALLOWED_URL_PREFIXES)})"
        )


def _ingest_dir(root: Path) -> RepoMap:
    repo = RepoMap(root=root)

    for current_root, dirs, filenames in os.walk(root):
        # Mutate dirs in place to skip uninteresting subtrees
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            full = Path(current_root) / filename
            rel = full.relative_to(root).as_posix()

            if full.suffix.lower() in _BINARY_EXTS:
                continue

            # A symlinked README/requirements pointing outside the repo would
            # otherwise leak host file contents into the LLM prompt.
            if full.is_symlink():
                continue

            try:
                size = full.stat().st_size
            except OSError:
                continue

            summary = FileSummary(path=rel, size_bytes=size)
            if full.suffix == ".py":
                _summarize_python(full, summary)
            repo.files.append(summary)

            if rel == "README.md" or rel == "readme.md":
                repo.readme = _read_text(full, limit_bytes=8 * 1024)
            elif rel == "pyproject.toml":
                repo.pyproject = _read_text(full, limit_bytes=8 * 1024)
            elif rel == "requirements.txt":
                repo.requirements = [
                    line.strip()
                    for line in _read_text(full).splitlines()
                    if line.strip() and not line.startswith("#")
                ]

    repo.files.sort(key=lambda f: f.path)
    return repo


def _read_text(path: Path, limit_bytes: int = 1_000_000) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(limit_bytes)
    except OSError:
        return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        return ""


def _summarize_python(path: Path, summary: FileSummary) -> None:
    text = _read_text(path)
    if not text:
        return
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                summary.classes.append(f"{node.name}({', '.join(methods[:6])})")
            else:
                summary.classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            summary.functions.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                summary.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            summary.imports.append(mod)
