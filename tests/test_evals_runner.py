"""Tests for evals/run.py — the corpus eval runner CLI."""

from __future__ import annotations

import json

import evals.run as run_mod
from evals.run import main


def _write_corpus(path):
    path.write_text("- id: sample\n  prompt: build a thing\n  rubrics: [compiles, tests_present]\n")


def test_offline_scores_existing_artifacts(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.yml"
    _write_corpus(corpus)

    artifacts = tmp_path / "artifacts" / "sample"
    artifacts.mkdir(parents=True)
    (artifacts / "final_program.py").write_text("def add(a, b):\n    return a + b\n")
    (artifacts / "test_program.py").write_text("def test_add():\n    assert True\n")

    monkeypatch.setattr(run_mod, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(run_mod, "REPORT_PATH", tmp_path / "report.json")

    exit_code = main(["--offline", "--corpus", str(corpus)])

    assert exit_code == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["pass_rate"] == 1.0
    assert report["cases"][0]["case_id"] == "sample"


def test_offline_failing_rubric_gives_nonzero_exit(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.yml"
    _write_corpus(corpus)

    artifacts = tmp_path / "artifacts" / "sample"
    artifacts.mkdir(parents=True)
    (artifacts / "final_program.py").write_text("def broken(:\n")  # syntax error

    monkeypatch.setattr(run_mod, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(run_mod, "REPORT_PATH", tmp_path / "report.json")

    exit_code = main(["--offline", "--corpus", str(corpus)])

    assert exit_code == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["pass_rate"] == 0.0


def test_unknown_case_id_exits_2(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.yml"
    _write_corpus(corpus)
    monkeypatch.setattr(run_mod, "REPORT_PATH", tmp_path / "report.json")

    assert main(["--offline", "--corpus", str(corpus), "--case", "nope"]) == 2


def test_offline_missing_artifacts_skips_case(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus.yml"
    _write_corpus(corpus)
    monkeypatch.setattr(run_mod, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(run_mod, "REPORT_PATH", tmp_path / "report.json")

    exit_code = main(["--offline", "--corpus", str(corpus)])

    assert exit_code == 1  # nothing scored -> pass rate 0
    assert "skipping" in capsys.readouterr().out
