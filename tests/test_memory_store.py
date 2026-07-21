"""T1.3: project + user memory stores."""

from __future__ import annotations

import json

import pytest

from core.memory_store import (
    DefaultVectorBackend,
    MemoryEntry,
    ProjectMemory,
    UserMemory,
    _cosine,
    _tokenize,
    project_id_for,
)

# ===========================================
# Tokenization + similarity
# ===========================================

class TestTokenization:
    def test_tokenize_lowercases_and_extracts_words(self):
        assert _tokenize("Build a CLI Tool!") == ["build", "a", "cli", "tool"]

    def test_tokenize_keeps_alphanumeric(self):
        assert _tokenize("python3 fastapi") == ["python3", "fastapi"]


class TestCosine:
    def test_identical_returns_1(self):
        a = {"x": 2, "y": 3}
        assert _cosine(a, a) == pytest.approx(1.0)

    def test_disjoint_returns_0(self):
        assert _cosine({"a": 1}, {"b": 1}) == 0.0

    def test_partial_overlap(self):
        score = _cosine({"a": 1, "b": 1}, {"a": 1, "c": 1})
        assert 0 < score < 1


# ===========================================
# DefaultVectorBackend
# ===========================================

class TestDefaultBackend:
    def test_add_and_query(self, tmp_path):
        backend = DefaultVectorBackend(str(tmp_path / "m.json"))
        backend.add(MemoryEntry(id="1", text="prefer fastapi over flask", kind="preference"))
        backend.add(MemoryEntry(id="2", text="use pytest for testing", kind="preference"))
        backend.add(MemoryEntry(id="3", text="default to sqlite for storage", kind="decision"))

        # Bag-of-words backend: query must share lexical tokens with the right entry
        results = backend.query("considering fastapi or flask", k=3)
        assert results[0].entry.id == "1"

    def test_query_filters_by_kind(self, tmp_path):
        backend = DefaultVectorBackend(str(tmp_path / "m.json"))
        backend.add(MemoryEntry(id="1", text="prefer fastapi", kind="preference"))
        backend.add(MemoryEntry(id="2", text="prefer fastapi", kind="decision"))

        results = backend.query("fastapi", k=5, kind="decision")
        assert len(results) == 1
        assert results[0].entry.kind == "decision"

    def test_persistence_round_trip(self, tmp_path):
        path = tmp_path / "m.json"
        b1 = DefaultVectorBackend(str(path))
        b1.add(MemoryEntry(id="x", text="hello world", kind="note"))

        b2 = DefaultVectorBackend(str(path))
        results = b2.query("hello", k=1)
        assert results and results[0].entry.id == "x"

    def test_save_format_is_inspectable_json(self, tmp_path):
        path = tmp_path / "m.json"
        backend = DefaultVectorBackend(str(path))
        backend.add(MemoryEntry(id="x", text="hello", kind="note", tags=["greeting"]))
        loaded = json.loads(path.read_text())
        assert "entries" in loaded
        assert loaded["entries"][0]["text"] == "hello"
        assert loaded["entries"][0]["tags"] == ["greeting"]

    def test_delete_and_clear(self, tmp_path):
        backend = DefaultVectorBackend(str(tmp_path / "m.json"))
        backend.add(MemoryEntry(id="1", text="a", kind="x"))
        backend.add(MemoryEntry(id="2", text="b", kind="x"))
        backend.delete("1")
        assert len(backend.all()) == 1

        backend.clear()
        assert backend.all() == []


# ===========================================
# UserMemory facade
# ===========================================

class TestUserMemory:
    def test_remember_and_recall(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MEMORY_BACKEND", raising=False)
        path = tmp_path / "user.json"

        # Inject path via subclass with explicit backend
        from core.memory_store import DefaultVectorBackend as _Backend

        mem = UserMemory.__new__(UserMemory)
        mem._backend = _Backend(str(path))

        mem.remember("Prefer FastAPI over Flask for new APIs", kind="preference")
        mem.remember("Always include pytest tests", kind="preference")
        mem.remember("Avoid heavyweight ORMs like SQLAlchemy", kind="preference")

        results = mem.recall("starting a new web API project", k=2)
        assert results
        # Top hit should be relevant
        text = " ".join(r.entry.text.lower() for r in results)
        assert "fastapi" in text or "api" in text

    def test_render_for_prompt_injection(self, tmp_path):
        from core.memory_store import DefaultVectorBackend as _Backend

        mem = UserMemory.__new__(UserMemory)
        mem._backend = _Backend(str(tmp_path / "u.json"))

        mem.remember("use type hints everywhere", kind="preference", tags=["style"])
        rendered = mem.render(query="must use type hints in this module", k=3)
        assert "[remembered]" in rendered
        assert "type hints" in rendered

    def test_render_empty_when_no_match(self, tmp_path):
        from core.memory_store import DefaultVectorBackend as _Backend

        mem = UserMemory.__new__(UserMemory)
        mem._backend = _Backend(str(tmp_path / "u.json"))

        # Nothing stored — render must be empty (not "[remembered]\n")
        rendered = mem.render("anything", k=3)
        assert rendered == ""


# ===========================================
# ProjectMemory
# ===========================================

class TestProjectMemory:
    def test_per_project_isolation(self, tmp_path):
        m1 = ProjectMemory("proj1", dir=str(tmp_path))
        m2 = ProjectMemory("proj2", dir=str(tmp_path))

        m1.remember("chose sqlite", kind="decision")
        results_in_2 = m2.recall("sqlite", k=5)
        assert results_in_2 == []

    def test_project_id_for_stable(self):
        a = project_id_for("Build a CLI todo app")
        b = project_id_for("Build a CLI todo app")
        c = project_id_for("Build a different thing")
        assert a == b
        assert a != c

    def test_project_id_strips_whitespace(self):
        assert project_id_for("hello") == project_id_for("  hello  ")


# ===========================================
# Backend factory
# ===========================================

class TestBackendFactory:
    def test_default_when_env_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEMORY_BACKEND", raising=False)
        from core.memory_store import make_backend

        backend = make_backend(str(tmp_path / "m.json"), "ignored")
        assert isinstance(backend, DefaultVectorBackend)

    def test_json_mode_explicit(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEMORY_BACKEND", "json")
        from core.memory_store import make_backend

        backend = make_backend(str(tmp_path / "m.json"), "ignored")
        assert isinstance(backend, DefaultVectorBackend)

    def test_chroma_mode_returns_chroma_backend(self, monkeypatch, tmp_path):
        # We don't import chromadb here; just verify the factory dispatches
        monkeypatch.setenv("MEMORY_BACKEND", "chroma")
        from core.memory_store import ChromaVectorBackend, make_backend

        backend = make_backend(str(tmp_path), "test_collection")
        assert isinstance(backend, ChromaVectorBackend)
        # Don't call methods that require chromadb installed
