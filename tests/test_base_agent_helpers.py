"""Tests for the shared LLM-reply parsing helpers in agents.base_agent."""

from __future__ import annotations

from agents.base_agent import extract_json, strip_code_fences


class TestStripCodeFences:
    def test_plain_code_untouched(self):
        assert strip_code_fences("x = 1\n") == "x = 1"

    def test_fenced_with_language_tag(self):
        assert strip_code_fences("```python\nx = 1\n```") == "x = 1"

    def test_fence_with_surrounding_prose(self):
        text = "Here you go:\n```python\nx = 1\n```\nHope that helps!"
        assert strip_code_fences(text) == "x = 1"

    def test_unclosed_fence(self):
        assert strip_code_fences("```python\nx = 1\n") == "x = 1"

    def test_code_containing_triple_backtick_string_survives(self):
        code = 'DOC = """usage: ```example```"""\nx = 1'
        # No leading fence: returned as-is
        assert strip_code_fences(code) == code


class TestExtractJson:
    def test_bare_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_trailing_prose(self):
        """The exact failure that used to collapse Architect designs."""
        text = '{"description": "d", "classes": {}}\n\nThis design keeps things simple.'
        assert extract_json(text) == {"description": "d", "classes": {}}

    def test_json_with_leading_and_trailing_prose(self):
        text = 'Sure! Here is the design:\n{"a": [1, 2]}\nLet me know.'
        assert extract_json(text) == {"a": [1, 2]}

    def test_list_payload(self):
        assert extract_json("Result:\n[1, 2, 3]\ndone") == [1, 2, 3]

    def test_garbage_returns_none(self):
        assert extract_json("no json here at all") is None


class TestSharedContextTopLevelOnly:
    def test_methods_not_reported_as_functions(self, tmp_path):
        from core.shared_context import SharedContext

        ctx = SharedContext(filepath=str(tmp_path / "ctx.json"))
        code = (
            "class TaskManager:\n    def add_task(self):\n        pass\n\ndef helper():\n    pass\n"
        )
        ctx.add_generated_code(1, "mod", code, "passed")
        assert ctx.get_defined_classes() == ["TaskManager"]
        # add_task is a method, not an importable top-level function
        assert ctx.get_defined_functions() == ["helper"]
