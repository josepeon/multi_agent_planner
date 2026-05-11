"""T3.3: language profile abstraction — enum, profile data, env resolution."""

from __future__ import annotations

import pytest

from core.languages import Language, LanguageProfile, get_profile


class TestLanguageEnum:
    def test_python_canonical(self):
        assert Language("python") is Language.PYTHON

    def test_typescript_canonical(self):
        assert Language("typescript") is Language.TYPESCRIPT

    @pytest.mark.parametrize("alias", ["py", "Python3", "PYTHON"])
    def test_python_aliases(self, alias):
        assert Language(alias) is Language.PYTHON

    @pytest.mark.parametrize("alias", ["ts", "node", "JavaScript", "js"])
    def test_typescript_aliases(self, alias):
        assert Language(alias) is Language.TYPESCRIPT

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            Language("rust")


class TestProfiles:
    def test_python_profile_shape(self):
        p = get_profile(Language.PYTHON)
        assert isinstance(p, LanguageProfile)
        assert p.file_extension == ".py"
        assert p.entrypoint == "main.py"
        assert p.test_framework == "pytest"
        assert p.comment_style == "#"

    def test_typescript_profile_shape(self):
        p = get_profile(Language.TYPESCRIPT)
        assert p.file_extension == ".ts"
        assert p.entrypoint == "main.ts"
        assert p.test_framework == "vitest"
        assert p.comment_style == "//"

    def test_file_for_appends_extension(self):
        p = get_profile(Language.PYTHON)
        assert p.file_for("main") == "main.py"

    def test_file_for_respects_existing_extension(self):
        p = get_profile(Language.PYTHON)
        assert p.file_for("README.md") == "README.md"

    def test_get_profile_accepts_string(self):
        p = get_profile("typescript")
        assert p.name is Language.TYPESCRIPT


class TestEnvResolution:
    def test_default_python(self, monkeypatch):
        monkeypatch.delenv("LANGUAGE", raising=False)
        assert get_profile().name is Language.PYTHON

    def test_env_typescript(self, monkeypatch):
        monkeypatch.setenv("LANGUAGE", "typescript")
        assert get_profile().name is Language.TYPESCRIPT

    def test_env_alias_recognized(self, monkeypatch):
        monkeypatch.setenv("LANGUAGE", "ts")
        assert get_profile().name is Language.TYPESCRIPT
