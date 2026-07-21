"""T3.2: eval harness — corpus loading, rubric scoring, report aggregation."""

from __future__ import annotations

import json

from core.eval_harness import (
    RUBRIC_REGISTRY,
    CaseResult,
    EvalCase,
    EvalReport,
    RubricContext,
    RubricResult,
    evaluate_artifacts,
    load_corpus,
    register_rubric,
    rubric_compiles,
    rubric_readme_nonempty,
    rubric_tests_pass,
    rubric_tests_present,
)

# ===========================================
# Corpus loader
# ===========================================


class TestLoadCorpus:
    def test_loads_minimal_yaml(self, tmp_path):
        path = tmp_path / "corpus.yml"
        path.write_text(
            "- id: case_a\n"
            "  prompt: Build something simple\n"
            "  rubrics: [compiles, tests_present]\n"
            "- id: case_b\n"
            "  prompt: Build something else\n"
            "  rubrics: [compiles]\n"
        )
        cases = load_corpus(str(path))
        assert len(cases) == 2
        assert cases[0].id == "case_a"
        assert "compiles" in cases[0].rubrics
        assert cases[1].id == "case_b"


# ===========================================
# Individual rubrics
# ===========================================


def _ctx(**kwargs):
    case = EvalCase(id="t", prompt="p", rubrics=[])
    defaults = {
        "case": case,
        "output_dir": kwargs.pop("output_dir", None) or __import__("pathlib").Path("/tmp"),
        "final_code": "",
        "test_code": "",
        "readme": "",
        "test_run": {},
    }
    defaults.update(kwargs)
    return RubricContext(**defaults)


class TestRubricCompiles:
    def test_passes_on_valid_code(self):
        result = rubric_compiles(_ctx(final_code="x = 1 + 2\n"))
        assert result.passed
        assert result.score == 1.0

    def test_fails_on_syntax_error(self):
        result = rubric_compiles(_ctx(final_code="def broken(:\n"))
        assert not result.passed
        assert "syntax" in result.detail.lower()

    def test_fails_on_empty(self):
        result = rubric_compiles(_ctx(final_code=""))
        assert not result.passed


class TestRubricTestsPresent:
    def test_passes_with_test_fn(self):
        result = rubric_tests_present(_ctx(test_code="def test_foo():\n    assert 1\n"))
        assert result.passed

    def test_passes_with_test_class(self):
        result = rubric_tests_present(
            _ctx(test_code="class TestFoo:\n    def test_x(self):\n        pass\n")
        )
        assert result.passed

    def test_fails_on_no_tests(self):
        result = rubric_tests_present(_ctx(test_code="x = 1\n"))
        assert not result.passed


class TestRubricTestsPass:
    def test_passes_when_all_passed(self):
        result = rubric_tests_pass(_ctx(test_run={"ran": True, "all_passed": True}))
        assert result.passed
        assert result.score == 1.0

    def test_partial_credit_on_some_failures(self):
        result = rubric_tests_pass(
            _ctx(test_run={"ran": True, "all_passed": False, "passed": 2, "total": 5})
        )
        assert not result.passed
        assert result.score == 0.4

    def test_fails_when_not_run(self):
        result = rubric_tests_pass(_ctx(test_run={"ran": False, "skip_reason": "no test file"}))
        assert not result.passed
        assert "skip_reason" not in result.detail  # the rendered detail uses our key
        assert "no test file" in result.detail


class TestRubricReadmeNonempty:
    def test_passes_when_long_enough(self):
        result = rubric_readme_nonempty(_ctx(readme="x" * 300))
        assert result.passed

    def test_partial_credit_short(self):
        result = rubric_readme_nonempty(_ctx(readme="x" * 100))
        assert not result.passed
        assert 0 < result.score < 1


class TestImportsRubric:
    def test_imports_click_passes(self):
        ctx = _ctx(final_code="import click\n@click.command()\ndef cli(): pass\n")
        result = RUBRIC_REGISTRY["imports_click"](ctx)
        assert result.passed

    def test_imports_click_fails(self):
        ctx = _ctx(final_code="print('hi')")
        result = RUBRIC_REGISTRY["imports_click"](ctx)
        assert not result.passed


class TestRegisterRubric:
    def test_custom_rubric_invocable(self, tmp_path):
        called = {"n": 0}

        def my_rubric(ctx):
            called["n"] += 1
            return RubricResult(name="my", passed=True, score=1.0)

        register_rubric("my_custom", my_rubric)
        case = EvalCase(id="x", prompt="p", rubrics=["my_custom"])
        result = evaluate_artifacts(case, tmp_path)
        assert called["n"] == 1
        assert result.aggregate_score == 1.0


# ===========================================
# Full evaluation
# ===========================================


class TestEvaluateArtifacts:
    def test_unknown_rubric_records_failure(self, tmp_path):
        case = EvalCase(id="x", prompt="p", rubrics=["doesnotexist"])
        result = evaluate_artifacts(case, tmp_path)
        assert not result.all_passed
        assert "unknown rubric" in result.rubrics[0].detail

    def test_reads_artifacts_from_disk(self, tmp_path):
        (tmp_path / "final_program.py").write_text("def f(): return 1\n")
        (tmp_path / "test_program.py").write_text("def test_f(): assert 1\n")
        (tmp_path / "README.md").write_text("# Project\n" + "info " * 100)
        (tmp_path / "test_results.json").write_text(
            json.dumps({"ran": True, "all_passed": True, "passed": 1, "total": 1})
        )

        case = EvalCase(
            id="x",
            prompt="p",
            rubrics=["compiles", "tests_present", "tests_pass", "readme_nonempty"],
        )
        result = evaluate_artifacts(case, tmp_path)
        assert result.all_passed
        assert result.aggregate_score == 1.0


# ===========================================
# Report aggregation
# ===========================================


class TestReport:
    def test_pass_rate_and_mean(self):
        report = EvalReport(
            cases=[
                CaseResult(
                    case_id="a",
                    rubrics=[RubricResult(name="r", passed=True, score=1.0)],
                    aggregate_score=1.0,
                ),
                CaseResult(
                    case_id="b",
                    rubrics=[RubricResult(name="r", passed=False, score=0.5)],
                    aggregate_score=0.5,
                ),
            ]
        )
        assert report.pass_rate() == 0.5
        assert report.mean_score() == 0.75

    def test_render_lists_each_case(self):
        report = EvalReport(
            cases=[
                CaseResult(
                    case_id="a",
                    rubrics=[RubricResult(name="r", passed=True, score=1.0)],
                    aggregate_score=1.0,
                ),
            ]
        )
        out = report.render()
        assert "a" in out
        assert "1.00" in out
