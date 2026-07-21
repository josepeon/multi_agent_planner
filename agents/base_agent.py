"""
Base Agent Module

Provides the abstract base class for all agents, plus the shared helpers for
parsing LLM replies (code-fence stripping, JSON extraction). Every agent used
to carry its own subtly-different copy of these; the weakest copies caused
real bugs (e.g. an architecture reply with a trailing sentence collapsed the
whole design to a text blob). One canonical implementation lives here.
"""

import json
import re
from typing import Any

from core.memory import Memory
from core.task_schema import Task

# Fences only count when they open a line — a ``` inside a string literal
# or docstring must not trigger extraction/truncation.
_FENCE_LINE_RE = re.compile(r"^```", re.MULTILINE)
_FENCED_BLOCK_RE = re.compile(r"^```[a-zA-Z0-9_+-]*[ \t]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_+-]*[ \t]*\n?", re.MULTILINE)


def strip_code_fences(text: str) -> str:
    """Extract code from an LLM reply that may wrap it in ``` fences.

    Handles a fenced block anywhere in the reply (leading/trailing prose),
    language tags, an unclosed trailing fence, and replies with no fences
    at all (returned stripped).
    """
    text = text.strip()
    if not _FENCE_LINE_RE.search(text):
        return text
    match = _FENCED_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    # Unclosed fence: drop everything up to and including the opening fence
    parts = _FENCE_OPEN_RE.split(text, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else text


def extract_json(text: str) -> Any:
    """Best-effort JSON payload from an LLM reply.

    Tries, in order: the reply as-is, the first fenced block, and the
    outermost {...} / [...] span (LLMs love appending "This design keeps
    things simple." after valid JSON). Returns None if nothing parses.
    """
    candidates = [text.strip(), strip_code_fences(text)]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


class BaseAgent:
    """Abstract base class for all agents."""

    name: str
    memory: Memory

    def __init__(self, name: str, memory_filepath: str | None = None) -> None:
        self.name = name
        self.memory = Memory(memory_filepath)

    def run(self, task: Task) -> Any:
        raise NotImplementedError(
            "Each agent must implement its own 'run' method for processing a Task object."
        )
