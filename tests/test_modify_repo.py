"""T2.6: modify-existing-codebase — repo ingestion + change-set parsing + apply."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.orchestrator_modify import (
    FileChange,
    ModificationPlan,
    _parse_changes,
)
from core.repo_ingestion import RepoMap, ingest

# ===========================================
# Ingestion
# ===========================================

class TestIngest:
    def test_local_dir(self, tmp_path):
        (tmp_path / "main.py").write_text(
            "def hello():\n    return 1\n\nclass Foo:\n    def bar(self):\n        return 2\n"
        )
        (tmp_path / "README.md").write_text("# Hello\n\nDocs here.\n")
        (tmp_path / "requirements.txt").write_text("flask\n# comment\npydantic>=2\n")

        repo = ingest(str(tmp_path))
        assert repo.file_count() == 3
        assert "# Hello" in repo.readme
        assert repo.requirements == ["flask", "pydantic>=2"]

        main = next(f for f in repo.files if f.path == "main.py")
        assert "hello" in main.functions
        assert any("Foo" in c for c in main.classes)

    def test_skips_binaries_and_hidden_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "image.png").write_bytes(b"\x89PNG fake")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("[core]\n")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"")

        repo = ingest(str(tmp_path))
        paths = {f.path for f in repo.files}
        assert "a.py" in paths
        assert "image.png" not in paths
        assert not any(p.startswith(".git") for p in paths)
        assert not any(p.startswith("__pycache__") for p in paths)

    def test_render_brief_shape(self, tmp_path):
        (tmp_path / "main.py").write_text(
            "class A:\n    def m(self): pass\n\ndef f(): pass\n"
        )
        (tmp_path / "README.md").write_text("# Project\n\nDesc.\n")
        brief = ingest(str(tmp_path)).render_brief()
        assert "1 file(s)" in brief or "2 file(s)" in brief or "3 file(s)" in brief
        assert "Python modules" in brief
        assert "main.py" in brief

    def test_nonexistent_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest(str(tmp_path / "does-not-exist"))

    def test_file_instead_of_dir_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            ingest(str(f))


# ===========================================
# Parsing change blocks
# ===========================================

class TestParseChanges:
    def _empty_repo(self, tmp_path) -> RepoMap:
        return RepoMap(root=Path(tmp_path))

    def test_parses_single_create(self, tmp_path):
        raw = (
            "--- FILE: new.py OP: create ---\n"
            "print('hi')\n"
        )
        changes = _parse_changes(self._empty_repo(tmp_path), raw)
        assert len(changes) == 1
        assert changes[0].operation == "create"
        assert changes[0].path == "new.py"
        assert "print" in changes[0].new_content

    def test_parses_multiple_files(self, tmp_path):
        raw = (
            "--- FILE: a.py OP: create ---\n"
            "a = 1\n"
            "\n"
            "--- FILE: b.py OP: modify ---\n"
            "b = 2\n"
        )
        changes = _parse_changes(self._empty_repo(tmp_path), raw)
        assert len(changes) == 2
        assert changes[0].path == "a.py" and changes[0].operation == "create"
        assert changes[1].path == "b.py" and changes[1].operation == "modify"

    def test_delete_has_no_content(self, tmp_path):
        raw = (
            "--- FILE: stale.py OP: delete ---\n"
        )
        changes = _parse_changes(self._empty_repo(tmp_path), raw)
        assert len(changes) == 1
        assert changes[0].operation == "delete"
        assert changes[0].new_content is None

    def test_ignores_text_before_first_header(self, tmp_path):
        raw = (
            "Some commentary the LLM should not have produced.\n"
            "--- FILE: ok.py OP: create ---\n"
            "x = 1\n"
        )
        changes = _parse_changes(self._empty_repo(tmp_path), raw)
        assert len(changes) == 1
        assert changes[0].path == "ok.py"


# ===========================================
# Apply
# ===========================================

class TestApply:
    def test_apply_dry_run_does_not_write(self, tmp_path):
        plan = ModificationPlan(
            repo_root=tmp_path,
            summary="",
            changes=[FileChange(operation="create", path="new.py", new_content="x = 1")],
        )
        result = plan.apply(dry_run=True)
        assert "new.py" in result["wrote"]
        assert result["dry_run"] is True
        # File not actually written
        assert not (tmp_path / "new.py").exists()

    def test_apply_writes_files(self, tmp_path):
        plan = ModificationPlan(
            repo_root=tmp_path,
            summary="",
            changes=[
                FileChange(operation="create", path="pkg/new.py", new_content="x = 1"),
                FileChange(operation="modify", path="existing.py", new_content="y = 2"),
            ],
        )
        result = plan.apply(dry_run=False)
        assert "pkg/new.py" in result["wrote"]
        assert (tmp_path / "pkg" / "new.py").read_text() == "x = 1"
        assert (tmp_path / "existing.py").read_text() == "y = 2"

    def test_apply_deletes(self, tmp_path):
        victim = tmp_path / "old.py"
        victim.write_text("# old\n")
        plan = ModificationPlan(
            repo_root=tmp_path,
            summary="",
            changes=[FileChange(operation="delete", path="old.py")],
        )
        plan.apply(dry_run=False)
        assert not victim.exists()

    def test_delete_missing_file_is_silent(self, tmp_path):
        plan = ModificationPlan(
            repo_root=tmp_path,
            summary="",
            changes=[FileChange(operation="delete", path="missing.py")],
        )
        # Should not raise
        plan.apply(dry_run=False)


# ===========================================
# Ingestion hardening
# ===========================================

class TestIngestionHardening:
    def test_ext_transport_rejected(self):
        """ext:: is a git transport that executes the embedded command."""
        import pytest

        from core.repo_ingestion import ingest
        with pytest.raises(ValueError, match="unsupported transport"):
            ingest("ext::sh -c 'touch /tmp/pwned' .git")

    def test_option_lookalike_rejected(self):
        import pytest

        from core.repo_ingestion import ingest
        with pytest.raises(ValueError, match="option"):
            ingest("--upload-pack=touch /tmp/pwned .git")

    def test_https_url_passes_validation(self):
        from core.repo_ingestion import _validate_git_url
        _validate_git_url("https://github.com/user/repo.git")  # no raise
        _validate_git_url("git@github.com:user/repo.git")  # no raise

    def test_symlinked_file_not_read(self, tmp_path):
        from core.repo_ingestion import ingest
        secret = tmp_path / "secret.txt"
        secret.write_text("HOST SECRET")
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "README.md").symlink_to(secret)
        (repo / "real.py").write_text("x = 1\n")

        repo_map = ingest(str(repo))
        assert "HOST SECRET" not in repo_map.readme
        assert any(f.path == "real.py" for f in repo_map.files)
