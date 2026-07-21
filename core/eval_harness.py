"""
Pipeline Eval Harness

The pipeline is hard to improve responsibly without a way to tell whether a
prompt tweak / model swap / routing change made things better or worse.
This module provides:

  - a YAML corpus of representative prompts ("build a CLI todo app", "build
    a small flask API", etc.)
  - automated rubric scoring on the pipeline's *artifacts* (no LLM call
    required for the baseline rubrics)
  - a runner that produces a regression report

The harness is deliberately *cheap* and *fast* in its default mode: rubrics
are static checks on the generated files (passes pytest run, syntactically
valid, expected imports present, etc.). A higher-quality LLM-as-judge mode
is available for the cases where rubrics aren't enough.

This is the pattern self-improving-agent uses (`evaluation/harness.py` with
golden-test YAML). We mirror its shape so future cross-repo reuse is clean.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ===========================================
# Spec types
# ===========================================

@dataclass
class EvalCase:
    """One row in the eval corpus."""

    id: str
    prompt: str
    rubrics: list[str]  # rubric names; resolved by RUBRIC_REGISTRY
    metadata: dict = field(default_factory=dict)


@dataclass
class RubricResult:
    name: str
    passed: bool
    score: float  # 0..1
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CaseResult:
    case_id: str
    rubrics: list[RubricResult] = field(default_factory=list)
    aggregate_score: float = 0.0
    notes: str = ""

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.rubrics)

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "rubrics": [r.as_dict() for r in self.rubrics],
            "aggregate_score": self.aggregate_score,
            "all_passed": self.all_passed,
            "notes": self.notes,
        }


@dataclass
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)

    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.all_passed) / len(self.cases)

    def mean_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.aggregate_score for c in self.cases) / len(self.cases)

    def as_dict(self) -> dict:
        return {
            "cases": [c.as_dict() for c in self.cases],
            "pass_rate": self.pass_rate(),
            "mean_score": self.mean_score(),
        }

    def render(self) -> str:
        lines = [
            f"Eval report — {len(self.cases)} case(s) — "
            f"pass rate {self.pass_rate():.0%}, mean score {self.mean_score():.2f}",
        ]
        for c in self.cases:
            tag = "OK" if c.all_passed else "--"
            lines.append(f"  [{tag}] {c.case_id}: score={c.aggregate_score:.2f}")
            for r in c.rubrics:
                marker = "PASS" if r.passed else "FAIL"
                detail = f" — {r.detail}" if r.detail else ""
                lines.append(f"      [{marker}] {r.name} ({r.score:.2f}){detail}")
        return "\n".join(lines)


# ===========================================
# Rubrics
# ===========================================

@dataclass
class RubricContext:
    """What every rubric receives."""

    case: EvalCase
    output_dir: Path
    final_code: str
    test_code: str
    readme: str
    test_run: dict  # output/test_results.json contents (if any)


RubricFn = Callable[[RubricContext], RubricResult]


def rubric_compiles(ctx: RubricContext) -> RubricResult:
    """Final code parses as valid Python."""
    if not ctx.final_code.strip():
        return RubricResult(
            name="compiles", passed=False, score=0.0,
            detail="no final_program.py produced"
        )
    try:
        ast.parse(ctx.final_code)
    except SyntaxError as exc:
        return RubricResult(
            name="compiles", passed=False, score=0.0,
            detail=f"syntax error at line {exc.lineno}: {exc.msg}"
        )
    return RubricResult(name="compiles", passed=True, score=1.0)


def rubric_tests_present(ctx: RubricContext) -> RubricResult:
    """Generated test file contains at least one test function."""
    if not ctx.test_code:
        return RubricResult(
            name="tests_present", passed=False, score=0.0,
            detail="no test_program.py produced"
        )
    has_test_fn = bool(re.search(r"^def test_\w", ctx.test_code, re.MULTILINE))
    has_test_class = bool(re.search(r"^class Test\w", ctx.test_code, re.MULTILINE))
    if has_test_fn or has_test_class:
        return RubricResult(name="tests_present", passed=True, score=1.0)
    return RubricResult(
        name="tests_present", passed=False, score=0.0,
        detail="no def test_* or class Test* found"
    )


def rubric_tests_pass(ctx: RubricContext) -> RubricResult:
    """T1's test-runner output records the generated tests passing."""
    if not ctx.test_run:
        return RubricResult(
            name="tests_pass", passed=False, score=0.0,
            detail="no test_results.json from pipeline"
        )
    if not ctx.test_run.get("ran"):
        return RubricResult(
            name="tests_pass", passed=False, score=0.0,
            detail=f"tests not run: {ctx.test_run.get('skip_reason', 'unknown')}"
        )
    if ctx.test_run.get("all_passed"):
        return RubricResult(name="tests_pass", passed=True, score=1.0)
    total = max(ctx.test_run.get("total", 0), 1)
    passed = ctx.test_run.get("passed", 0)
    return RubricResult(
        name="tests_pass",
        passed=False,
        score=passed / total,
        detail=f"{passed}/{total} passed",
    )


def rubric_readme_nonempty(ctx: RubricContext) -> RubricResult:
    """README has at least 200 characters."""
    n = len(ctx.readme.strip())
    if n >= 200:
        return RubricResult(name="readme_nonempty", passed=True, score=1.0)
    return RubricResult(
        name="readme_nonempty", passed=False, score=n / 200,
        detail=f"only {n} chars"
    )


def _make_imports_rubric(*expected: str) -> RubricFn:
    """Generate a rubric that checks the final code imports a set of modules."""
    name = "imports_" + "_".join(expected)

    def rubric(ctx: RubricContext) -> RubricResult:
        present = [imp for imp in expected if re.search(rf"\b{imp}\b", ctx.final_code)]
        score = len(present) / max(len(expected), 1)
        return RubricResult(
            name=name,
            passed=score >= 1.0,
            score=score,
            detail=f"found {sorted(present)} / expected {list(expected)}",
        )

    rubric.__name__ = name
    return rubric


RUBRIC_REGISTRY: dict[str, RubricFn] = {
    "compiles": rubric_compiles,
    "tests_present": rubric_tests_present,
    "tests_pass": rubric_tests_pass,
    "readme_nonempty": rubric_readme_nonempty,
    # A few common project-shape rubrics
    "imports_click": _make_imports_rubric("click"),
    "imports_flask": _make_imports_rubric("flask"),
    "imports_fastapi": _make_imports_rubric("fastapi"),
    "imports_pytest": _make_imports_rubric("pytest"),
    "imports_dataclasses": _make_imports_rubric("dataclasses"),
}


def register_rubric(name: str, fn: RubricFn) -> None:
    """Add a custom rubric to the registry."""
    RUBRIC_REGISTRY[name] = fn


# ===========================================
# Corpus loader (light YAML)
# ===========================================

def load_corpus(path: str | Path) -> list[EvalCase]:
    """Load an eval corpus from a tiny YAML file.

    Format:
        - id: cli_todo
          prompt: Build a CLI todo app with click
          rubrics: [compiles, tests_present, tests_pass, imports_click]
        - id: flask_api
          ...

    We keep parsing simple to avoid a PyYAML dep. Only top-level lists of
    objects with these four keys (id / prompt / rubrics / metadata) are
    supported. For richer formats, install PyYAML and we'll use it.
    """
    text = Path(path).read_text()
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(text)
        return [EvalCase(**item) for item in raw]
    except ImportError:
        pass
    return _parse_minimal_yaml(text)


def _parse_minimal_yaml(text: str) -> list[EvalCase]:
    """Hand-rolled parser for the limited shape above.

    Each case starts with '- id:'. We collect lines until the next '- id:'
    or EOF; within a case, look for 'id:', 'prompt:', 'rubrics:'.
    """
    cases: list[EvalCase] = []
    current: dict | None = None

    def flush():
        if not current:
            return
        cases.append(EvalCase(
            id=str(current.get("id", "")),
            prompt=str(current.get("prompt", "")),
            rubrics=current.get("rubrics", []),
            metadata=current.get("metadata", {}),
        ))

    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("- id:"):
            flush()
            current = {"id": stripped.split(":", 1)[1].strip()}
            continue
        if current is None:
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.lstrip().partition(":")
        key = key.strip()
        value = value.strip()
        if key == "id":
            current["id"] = value
        elif key == "prompt":
            current["prompt"] = value
        elif key == "rubrics":
            # Expect inline list: [a, b, c]
            value = value.strip("[]")
            current["rubrics"] = [
                v.strip() for v in value.split(",") if v.strip()
            ]
    flush()
    return cases


# ===========================================
# Runner
# ===========================================

def evaluate_artifacts(
    case: EvalCase,
    output_dir: str | Path,
) -> CaseResult:
    """Apply every rubric in ``case`` against the files in ``output_dir``."""
    output = Path(output_dir)
    ctx = RubricContext(
        case=case,
        output_dir=output,
        final_code=_read_safe(output / "final_program.py"),
        test_code=_read_safe(output / "test_program.py"),
        readme=_read_safe(output / "README.md"),
        test_run=_read_json_safe(output / "test_results.json"),
    )

    rubric_results: list[RubricResult] = []
    for rubric_name in case.rubrics:
        fn = RUBRIC_REGISTRY.get(rubric_name)
        if fn is None:
            rubric_results.append(RubricResult(
                name=rubric_name, passed=False, score=0.0,
                detail=f"unknown rubric '{rubric_name}'"
            ))
            continue
        try:
            rubric_results.append(fn(ctx))
        except Exception as exc:  # noqa: BLE001
            rubric_results.append(RubricResult(
                name=rubric_name, passed=False, score=0.0,
                detail=f"rubric raised: {type(exc).__name__}: {exc}"
            ))

    aggregate = (
        sum(r.score for r in rubric_results) / len(rubric_results)
        if rubric_results else 0.0
    )
    return CaseResult(
        case_id=case.id,
        rubrics=rubric_results,
        aggregate_score=aggregate,
    )


def _read_safe(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _read_json_safe(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return {}
    except json.JSONDecodeError:
        return {}
