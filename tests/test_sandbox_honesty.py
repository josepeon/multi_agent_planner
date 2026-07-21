"""B2: subprocess sandbox is honest about what it provides.

The legacy mode name 'subprocess' implied security isolation it never had.
These tests verify:

1. The canonical name is 'crash_isolated'.
2. The legacy 'subprocess' name is accepted but emits a DeprecationWarning.
3. MAP_FORBID_CRASH_ISOLATED=1 disables both direct invocation and the
   restricted-mode fallback for input/GUI/eval code.
"""

from __future__ import annotations

import warnings

import pytest

from core.sandbox import (
    FORBID_CRASH_ISOLATED_ENV,
    ExecutionMethod,
    crash_isolated_forbidden,
    execute_code_safely,
)

# ===========================================
# Canonical name + alias
# ===========================================


class TestNaming:
    def test_canonical_name_resolves(self):
        assert ExecutionMethod("crash_isolated") is ExecutionMethod.CRASH_ISOLATED

    def test_legacy_subprocess_alias_works_with_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            method = ExecutionMethod("subprocess")
        assert method is ExecutionMethod.CRASH_ISOLATED
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("crash_isolated" in str(w.message) for w in deprecations)


# ===========================================
# Forbid flag
# ===========================================


class TestForbidFlag:
    def test_forbidden_when_env_set(self, monkeypatch):
        monkeypatch.setenv(FORBID_CRASH_ISOLATED_ENV, "1")
        assert crash_isolated_forbidden() is True

    def test_not_forbidden_by_default(self, monkeypatch):
        monkeypatch.delenv(FORBID_CRASH_ISOLATED_ENV, raising=False)
        assert crash_isolated_forbidden() is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv(FORBID_CRASH_ISOLATED_ENV, val)
        assert crash_isolated_forbidden() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv(FORBID_CRASH_ISOLATED_ENV, val)
        assert crash_isolated_forbidden() is False


# ===========================================
# Behavior: forbid flag short-circuits
# ===========================================


class TestForbidBehavior:
    def test_direct_crash_isolated_call_is_blocked(self, monkeypatch):
        monkeypatch.setenv(FORBID_CRASH_ISOLATED_ENV, "1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = execute_code_safely("print('hello')", method="subprocess")
        assert result["success"] is False
        assert result["method_used"] == "crash_isolated_forbidden"
        assert FORBID_CRASH_ISOLATED_ENV in result["error"]

    def test_restricted_fallback_blocked_for_input_code(self, monkeypatch):
        monkeypatch.setenv(FORBID_CRASH_ISOLATED_ENV, "1")
        # Code that would normally trigger the subprocess fallback path
        code = "x = input('name? ')\nprint(x)"
        result = execute_code_safely(code, method="restricted")
        assert result["success"] is False
        assert result["method_used"] == "crash_isolated_forbidden"

    def test_restricted_fallback_path_not_blocked_when_not_forbidden(self, monkeypatch):
        monkeypatch.delenv(FORBID_CRASH_ISOLATED_ENV, raising=False)
        # Code with input() in a comment triggers the fallback heuristic. The
        # subprocess executor then short-circuits with a "skipping" notice for
        # input/GUI code. The point of this test: the forbid path is NOT taken.
        code = "# uses input() in a comment\nprint('ok')"
        result = execute_code_safely(code, method="restricted")
        assert result["method_used"] != "crash_isolated_forbidden"

    def test_restricted_path_unaffected_by_flag_when_no_fallback_needed(self, monkeypatch):
        # Simple code that runs in pure restricted mode shouldn't be affected
        # even when the forbid flag is set.
        monkeypatch.setenv(FORBID_CRASH_ISOLATED_ENV, "1")
        result = execute_code_safely("x = 1 + 1\nprint(x)", method="restricted")
        assert result["success"] is True
        assert "2" in result["output"]
