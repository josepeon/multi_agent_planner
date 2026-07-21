"""
Test Runner Module

Executes generated pytest tests against generated code in a temp directory and
reports the result. Closes the loop on the previously-broken claim that tests
were verified after generation.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestRunResult:
    """Outcome of running generated tests against generated code."""

    ran: bool
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total: int = 0
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    failure_summaries: list[str] = field(default_factory=list)
    skip_reason: str | None = None

    @property
    def all_passed(self) -> bool:
        return self.ran and self.failed == 0 and self.errors == 0 and self.total > 0

    def as_dict(self) -> dict:
        return {
            "ran": self.ran,
            "all_passed": self.all_passed,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "total": self.total,
            "duration_seconds": self.duration_seconds,
            "skip_reason": self.skip_reason,
            "failure_summaries": self.failure_summaries,
        }


_FAILURE_PATTERN = re.compile(r"FAILED (.+?)(?:\s-\s(.+))?$", re.MULTILINE)
_SUMMARY_PATTERN = re.compile(
    r"(?:=+\s+)?"
    r"(?:(?P<passed>\d+) passed)?"
    r"(?:[, ]+(?P<failed>\d+) failed)?"
    r"(?:[, ]+(?P<errors>\d+) errors?)?"
    r"(?:[, ]+(?P<skipped>\d+) skipped)?"
    r"(?:[, ]+\d+ warnings?)?"
    r"(?: in (?P<duration>[\d.]+)s)",
)


def _parse_pytest_output(stdout: str) -> dict:
    """Best-effort parse of pytest's terminal summary line.

    pytest doesn't have a stable JSON output without a plugin, so we parse the
    last summary line. We accept partial matches; missing groups stay zero.
    """
    last_match = None
    for match in _SUMMARY_PATTERN.finditer(stdout):
        last_match = match

    if not last_match:
        return {}

    groups = last_match.groupdict()
    return {
        "passed": int(groups.get("passed") or 0),
        "failed": int(groups.get("failed") or 0),
        "errors": int(groups.get("errors") or 0),
        "skipped": int(groups.get("skipped") or 0),
        "duration_seconds": float(groups.get("duration") or 0.0),
    }


def _looks_runnable(test_code: str) -> tuple[bool, str | None]:
    """Cheap pre-checks to avoid spinning up a subprocess for unusable input."""
    if not test_code or not test_code.strip():
        return False, "empty test code"
    stripped = test_code.lstrip()
    if stripped.startswith("# Test generation failed"):
        return False, "upstream test generation failed"
    if "def test_" not in test_code and "class Test" not in test_code:
        return False, "no pytest test functions or classes detected"
    return True, None


_FROM_IMPORT_RE = re.compile(
    r"^from\s+([a-zA-Z_][\w]*)\s+import\b", re.MULTILINE
)
# 'import calculator' / 'import calculator as calc' — same aliasing need
_PLAIN_IMPORT_RE = re.compile(
    r"^import\s+([a-zA-Z_][\w]*)(?:\s+as\s+\w+)?\s*$", re.MULTILINE
)


def _extract_imported_module_names(test_code: str) -> set[str]:
    """Pull module names out of ``from X import ...`` and ``import X`` lines.

    These are the names the test code expects to find on disk. The test
    generator sometimes picks a project-specific name like ``calculator``
    rather than ``final_program``; the runner needs to expose the program
    under all of them.
    """
    names = {match.group(1) for match in _FROM_IMPORT_RE.finditer(test_code)}
    names |= {match.group(1) for match in _PLAIN_IMPORT_RE.finditer(test_code)}
    return names


# Names we never alias the program under (would shadow real packages
# the tests legitimately import).
_RESERVED_MODULE_NAMES = {
    "pytest", "os", "sys", "json", "math", "random", "datetime", "typing",
    "pathlib", "collections", "itertools", "functools", "abc",
    "dataclasses", "enum", "re", "string", "unittest",
}


def run_generated_tests(
    program_code: str,
    test_code: str,
    *,
    program_filename: str = "final_program.py",
    timeout_seconds: int = 60,
) -> TestRunResult:
    """Execute pytest on `test_code` against `program_code` in a temp dir.

    The program is written under ``program_filename`` (stem used as module
    name) so tests can ``from final_program import ...``. To survive cases
    where the test generator picked a different module name (e.g. ``from
    calculator import ...``), we *also* expose the same content under every
    name the test imports from — so tests don't fail with ModuleNotFoundError
    just because of an inconsistent name choice upstream.

    The function never raises on test failure — it returns a structured result.
    Genuine infrastructure failures (timeout, pytest missing) set `ran=True`
    with `errors > 0` and a populated stderr.
    """
    runnable, skip_reason = _looks_runnable(test_code)
    if not runnable:
        return TestRunResult(ran=False, skip_reason=skip_reason)

    module_stem = Path(program_filename).stem

    # Aliases: names the test code imports from but don't match program_filename.
    extra_modules = (
        _extract_imported_module_names(test_code)
        - {module_stem}
        - _RESERVED_MODULE_NAMES
    )

    with tempfile.TemporaryDirectory(prefix="map_test_run_") as tmp_str:
        tmp = Path(tmp_str)
        (tmp / program_filename).write_text(program_code)
        # Alias the same content under each alternate module name. Using
        # write rather than symlink keeps it portable across OS / filesystem.
        for alias in extra_modules:
            (tmp / f"{alias}.py").write_text(program_code)
        (tmp / "test_program.py").write_text(test_code)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-x",  # stop after first failure to keep output small
            "--tb=short",
            "-rN",  # no extra summary noise
            "test_program.py",
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={
                    **_inherit_env(),
                    "PYTHONPATH": str(tmp),
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                },
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return TestRunResult(
                ran=True,
                errors=1,
                total=1,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "")
                + f"\nTest run timed out after {timeout_seconds}s.",
                failure_summaries=[f"timeout after {timeout_seconds}s"],
            )
        except FileNotFoundError as exc:
            return TestRunResult(
                ran=False,
                skip_reason=f"could not invoke pytest: {exc}",
            )

        parsed = _parse_pytest_output(proc.stdout)
        failure_summaries = [
            f"{path}{(' — ' + reason) if reason else ''}"
            for path, reason in _FAILURE_PATTERN.findall(proc.stdout)
        ]

        passed = parsed.get("passed", 0)
        failed = parsed.get("failed", 0)
        errors = parsed.get("errors", 0)
        skipped = parsed.get("skipped", 0)
        total = passed + failed + errors + skipped

        # If pytest exited non-zero but our parser didn't find a summary
        # (e.g. collection error), surface it as one error.
        if total == 0 and proc.returncode != 0:
            errors = 1
            total = 1

        return TestRunResult(
            ran=True,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            total=total,
            duration_seconds=parsed.get("duration_seconds", 0.0),
            stdout=proc.stdout,
            stderr=proc.stderr,
            failure_summaries=failure_summaries,
        )


def _inherit_env() -> dict[str, str]:
    """Subset of the parent env safe to inherit into the test subprocess."""
    import os

    keep = {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR"}
    return {k: v for k, v in os.environ.items() if k in keep}


def render_summary(result: TestRunResult) -> str:
    """Human-readable one-block summary suitable for logs and READMEs."""
    if not result.ran:
        return f"Tests not executed: {result.skip_reason or 'unknown reason'}."
    if result.all_passed:
        return (
            f"Tests passed: {result.passed}/{result.total} "
            f"in {result.duration_seconds:.2f}s."
        )
    parts = [
        f"Tests: {result.passed} passed",
    ]
    if result.failed:
        parts.append(f"{result.failed} failed")
    if result.errors:
        parts.append(f"{result.errors} errors")
    if result.skipped:
        parts.append(f"{result.skipped} skipped")
    summary = ", ".join(parts) + f" (total {result.total})"
    if result.failure_summaries:
        summary += "\nFailures:\n  - " + "\n  - ".join(
            result.failure_summaries[:10]
        )
    return summary


def write_result_log(result: TestRunResult, path: str) -> None:
    """Persist the structured test result for downstream consumers."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.as_dict(), f, indent=2)
