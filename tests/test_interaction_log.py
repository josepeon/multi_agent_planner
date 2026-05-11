"""T3.4: interaction log + SIA export."""

from __future__ import annotations

import json

import pytest

from core import interaction_log as il
from core.interaction_log import (
    Interaction,
    InteractionLog,
    disable_logging,
    enable_logging,
    export_for_sia,
    record,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Each test starts with a fresh module-level log pointed at tmp_path."""
    new_log = InteractionLog(path=None)
    monkeypatch.setattr(il, "_log", new_log)
    yield


class TestInteractionLog:
    def test_disabled_by_default(self):
        assert il.get_log().enabled is False

    def test_record_is_noop_when_disabled(self, tmp_path):
        record(
            role="planner", provider="groq", model="x",
            system_message="s", user_message="u", response="r",
        )
        # No path means nothing should be written anywhere
        assert il.get_log().path is None

    def test_record_writes_when_enabled(self, tmp_path):
        log_path = tmp_path / "interactions.jsonl"
        enable_logging(str(log_path))
        record(
            role="planner", provider="groq", model="llama-3.3-70b-versatile",
            system_message="you are a planner",
            user_message="build a calculator",
            response="1. data class\n2. operations",
            prompt_tokens=50, completion_tokens=10,
        )
        assert log_path.exists()
        with open(log_path) as f:
            row = json.loads(f.readline())
        assert row["role"] == "planner"
        assert row["prompt_tokens"] == 50

    def test_multiple_records_append(self, tmp_path):
        enable_logging(str(tmp_path / "i.jsonl"))
        for i in range(3):
            record(
                role="developer", provider="groq", model="m",
                system_message="s", user_message=f"u{i}", response=f"r{i}",
            )
        rows = il.get_log().read_all()
        assert len(rows) == 3
        assert rows[1].user_message == "u1"

    def test_disable_stops_writing(self, tmp_path):
        path = tmp_path / "i.jsonl"
        enable_logging(str(path))
        record(role="x", provider="p", model="m", system_message="",
               user_message="u1", response="r1")
        disable_logging()
        record(role="x", provider="p", model="m", system_message="",
               user_message="u2", response="r2")
        rows = InteractionLog(path).read_all()
        assert len(rows) == 1
        assert rows[0].user_message == "u1"


class TestExportForSia:
    def test_writes_sia_format(self, tmp_path):
        enable_logging(str(tmp_path / "i.jsonl"))
        record(
            role="developer", provider="groq", model="m",
            system_message="you are dev",
            user_message="write add()",
            response="def add(a, b):\n    return a + b\n",
            prompt_tokens=20, completion_tokens=15,
        )
        out = tmp_path / "developer.jsonl"
        count = export_for_sia("developer", out)
        assert count == 1

        row = json.loads(out.read_text())
        msgs = row["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert "def add" in msgs[2]["content"]
        assert row["metadata"]["model"] == "m"

    def test_filters_by_role(self, tmp_path):
        enable_logging(str(tmp_path / "i.jsonl"))
        record(role="planner", provider="g", model="m",
               system_message="", user_message="x",
               response="A reasonable planner output that exceeds the min_response_chars filter")
        record(role="developer", provider="g", model="m",
               system_message="", user_message="x",
               response="A reasonable developer output that exceeds the min_response_chars filter")

        out = tmp_path / "planner.jsonl"
        count = export_for_sia("planner", out)
        assert count == 1
        row = json.loads(out.read_text())
        assert "planner output" in row["messages"][2]["content"]

    def test_filters_short_responses(self, tmp_path):
        enable_logging(str(tmp_path / "i.jsonl"))
        record(role="x", provider="g", model="m",
               system_message="", user_message="u", response="ok")
        record(role="x", provider="g", model="m",
               system_message="", user_message="u", response="x" * 50)

        out = tmp_path / "x.jsonl"
        count = export_for_sia("x", out, min_response_chars=10)
        assert count == 1

    def test_no_matching_rows_writes_empty_file(self, tmp_path):
        enable_logging(str(tmp_path / "i.jsonl"))
        record(role="planner", provider="g", model="m",
               system_message="", user_message="u", response="some content here")

        out = tmp_path / "developer.jsonl"
        count = export_for_sia("developer", out)
        assert count == 0
        assert out.exists()
        assert out.read_text() == ""
