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

_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Extract code from an LLM reply that may wrap it in ``` fences.

    Handles a fenced block anywhere in the reply (leading/trailing prose),
    language tags, an unclosed trailing fence, and replies with no fences
    at all (returned stripped).
    """
    text = text.strip()
    if "```" not in text:
        return text
    match = _FENCED_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    # Unclosed fence: drop the opening fence line, keep the rest
    after = text.split("```", 1)[1]
    if "\n" in after:
        after = after.split("\n", 1)[1]
    else:
        # ```json style tag with no newline — nothing usable before EOL
        after = ""
    return after.strip()


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
        raise NotImplementedError("Each agent must implement its own 'run' method for processing a Task object.")
