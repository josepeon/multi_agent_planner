"""Tests for core.test_runner — verifies B1 (run generated tests) actually runs."""

from __future__ import annotations

import json

from core.test_runner import (
    TestRunResult,
    render_summary,
    run_generated_tests,
    write_result_log,
)

# ===========================================
# Skip conditions
# ===========================================


class TestSkipConditions:
    def test_empty_test_code_does_not_run(self):
        result = run_generated_tests("def add(a, b): return a + b", "")
        assert result.ran is False
        assert result.skip_reason

    def test_failed_marker_does_not_run(self):
        result = run_generated_tests(
            "x = 1",
            "# Test generation failed: rate limit",
        )
        assert result.ran is False
        assert "failed" in result.skip_reason

    def test_no_test_functions_does_not_run(self):
        result = run_generated_tests(
            "x = 1",
            "import math\nx = math.pi",
        )
        assert result.ran is False
        assert "test functions" in result.skip_reason


# ===========================================
# Real execution paths
# ===========================================


class TestRealExecution:
    def test_passing_tests_report_success(self):
        program = "def add(a, b):\n    return a + b\n"
        tests = (
            "from final_program import add\n\n"
            "def test_add_positive():\n"
            "    assert add(2, 3) == 5\n\n"
            "def test_add_negative():\n"
            "    assert add(-1, -1) == -2\n"
        )
        result = run_generated_tests(program, tests)
        assert result.ran is True
        assert result.all_passed is True
        assert result.passed == 2
        assert result.failed == 0
        assert result.total == 2

    def test_failing_test_is_reported(self):
        program = "def add(a, b):\n    return a + b\n"
        tests = (
            "from final_program import add\n\ndef test_add_wrong():\n    assert add(2, 3) == 99\n"
        )
        result = run_generated_tests(program, tests)
        assert result.ran is True
        assert result.all_passed is False
        assert result.failed >= 1

    def test_aliases_program_under_imported_module_name(self):
        """When the test generator imports from a project-specific name
        (e.g. 'calculator') instead of 'final_program', the runner should
        still find the code. Mirrors the real bug surfaced in the MAP
        smoke test."""
        program = "def add(a, b):\n    return a + b\n"
        tests = (
            "from calculator import add\n\ndef test_add_positive():\n    assert add(2, 3) == 5\n"
        )
        result = run_generated_tests(program, tests)
        assert result.ran is True
        assert result.all_passed is True
        assert result.passed == 1

    def test_multiple_import_names_all_aliased(self):
        """If tests import from several names, all of them resolve."""
        program = "def add(a, b):\n    return a + b\nVALUE = 42\n"
        tests = (
            "from calculator import add\n"
            "from final_program import VALUE\n\n"
            "def test_both():\n"
            "    assert add(VALUE, 1) == 43\n"
        )
        result = run_generated_tests(program, tests)
        assert result.ran is True
        assert result.all_passed is True

    def test_reserved_module_names_not_aliased(self):
        """Aliasing must not shadow standard library imports."""
        # 'json' is in the reserved list; an 'import json' in test_code
        # should NOT cause us to clobber the real json module.
        program = "def add(a, b):\n    return a + b\n"
        tests = (
            "import json\n"
            "from final_program import add\n\n"
            "def test_uses_json():\n"
            "    payload = json.dumps({'sum': add(1, 2)})\n"
            "    assert json.loads(payload) == {'sum': 3}\n"
        )
        result = run_generated_tests(program, tests)
        assert result.ran is True
        assert result.all_passed is True

    def test_collection_error_surfaces_as_error(self):
        program = "def add(a, b):\n    return a + b\n"
        # 'os' is a reserved module name (never aliased to the program), so a
        # bogus from-import on it produces a genuine collection error.
        tests = "from os import no_such_name\n\ndef test_x():\n    assert True\n"
        result = run_generated_tests(program, tests)
        assert result.ran is True
        assert result.all_passed is False


# ===========================================
# Reporting helpers
# ===========================================


class TestReporting:
    def test_render_summary_for_skip(self):
        result = TestRunResult(ran=False, skip_reason="empty test code")
        out = render_summary(result)
        assert "not executed" in out
        assert "empty test code" in out

    def test_render_summary_for_pass(self):
        result = TestRunResult(ran=True, passed=3, total=3, duration_seconds=0.42)
        out = render_summary(result)
        assert "3/3" in out

    def test_render_summary_for_fail_includes_failures(self):
        result = TestRunResult(
            ran=True,
            passed=1,
            failed=2,
            total=3,
            failure_summaries=["test_program.py::test_a", "test_program.py::test_b"],
        )
        out = render_summary(result)
        assert "1 passed" in out
        assert "2 failed" in out
        assert "test_a" in out

    def test_write_result_log_persists_json(self, tmp_path):
        result = TestRunResult(ran=True, passed=2, total=2)
        path = tmp_path / "nested" / "result.json"
        write_result_log(result, str(path))
        loaded = json.loads(path.read_text())
        assert loaded["all_passed"] is True
        assert loaded["passed"] == 2

    def test_as_dict_round_trip_keys(self):
        result = TestRunResult(ran=True, passed=1, failed=1, total=2)
        d = result.as_dict()
        for key in ("ran", "all_passed", "passed", "failed", "errors", "skipped", "total"):
            assert key in d


class TestPlainImportAliasing:
    def test_plain_import_gets_aliased(self):
        """`import calculator` (not just `from calculator import x`) must
        resolve to the generated program."""
        program = "def add(a, b):\n    return a + b\n"
        tests = "import calculator\n\ndef test_add():\n    assert calculator.add(1, 2) == 3\n"
        result = run_generated_tests(program, tests)
        assert result.ran
        assert result.all_passed, result.failure_summaries

    def test_import_as_alias_form(self):
        program = "def add(a, b):\n    return a + b\n"
        tests = "import calculator as calc\n\ndef test_add():\n    assert calc.add(2, 2) == 4\n"
        result = run_generated_tests(program, tests)
        assert result.ran
        assert result.all_passed, result.failure_summaries
