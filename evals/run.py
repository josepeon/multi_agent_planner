"""Eval runner: score the pipeline against the corpus in evals/corpus.yml.

Two modes:

  Full (default) — runs the real pipeline once per corpus case, snapshots the
  artifacts into evals/artifacts/<case_id>/, and scores them. Needs an LLM
  API key and spends real tokens:

      python -m evals.run
      python -m evals.run --case cli_todo_click        # single case
      python -m evals.run --corpus path/to/other.yml

  Offline — re-scores whatever is already in evals/artifacts/ without
  touching an LLM. Free; useful after changing a rubric:

      python -m evals.run --offline

The report prints to stdout and is written to evals/report.json so two runs
can be diffed for regressions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Ensure project root on the path so `from core...` works when invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval_harness import (
    EvalCase,
    EvalReport,
    evaluate_artifacts,
    load_corpus,
)

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = EVALS_DIR / "corpus.yml"
ARTIFACTS_DIR = EVALS_DIR / "artifacts"
REPORT_PATH = EVALS_DIR / "report.json"

# The pipeline writes these to output/; we snapshot them per case.
ARTIFACT_FILES = (
    "final_program.py",
    "test_program.py",
    "README.md",
    "test_results.json",
)


def _snapshot_output(case_id: str, output_dir: Path = Path("output")) -> Path:
    """Copy the pipeline's artifacts for one case into evals/artifacts/<id>/."""
    dest = ARTIFACTS_DIR / case_id
    dest.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
    return dest


def _run_case_pipeline(case: EvalCase) -> Path:
    """Run the real pipeline for one corpus case; return its artifact dir."""
    # Imported lazily: pulls in every agent + needs an API key, which the
    # offline mode should never require.
    from core.orchestrator import run_pipeline
    from core.task_schema import Task

    # Clear stale artifacts so a failed run can't be scored on the previous
    # case's files.
    output_dir = Path("output")
    for name in ARTIFACT_FILES:
        (output_dir / name).unlink(missing_ok=True)

    task = Task(id=int(time.time()), description=case.prompt)
    try:
        run_pipeline(task, save_path=f"output/session_log_{case.id}.json")
    except Exception as exc:  # noqa: BLE001 — a crashed case scores 0, run continues
        print(f"  !! pipeline crashed on '{case.id}': {type(exc).__name__}: {exc}")
    return _snapshot_output(case.id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Corpus YAML path.")
    parser.add_argument("--case", default=None, help="Run only this case id.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Score existing evals/artifacts/ without running the pipeline.",
    )
    args = parser.parse_args(argv)

    cases = load_corpus(args.corpus)
    if args.case:
        cases = [c for c in cases if c.id == args.case]
        if not cases:
            print(f"No case '{args.case}' in {args.corpus}")
            return 2

    report = EvalReport()
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.id}")
        if args.offline:
            artifact_dir = ARTIFACTS_DIR / case.id
            if not artifact_dir.exists():
                print("  (no artifacts; skipping — run without --offline first)")
                continue
        else:
            artifact_dir = _run_case_pipeline(case)
        report.cases.append(evaluate_artifacts(case, artifact_dir))

    print()
    print(report.render())
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2))
    print(f"\nReport written to {REPORT_PATH}")
    return 0 if report.pass_rate() == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
