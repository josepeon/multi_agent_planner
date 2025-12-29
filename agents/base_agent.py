"""
Base Agent Module

Provides the abstract base class for all agents in the multi-agent system.
"""

from typing import Any

from core.memory import Memory
from core.task_schema import Task


class BaseAgent:
    """Abstract base class for all agents."""

    name: str
    memory: Memory

    def __init__(self, name: str, memory_filepath: str | None = None) -> None:
        self.name = name
        self.memory = Memory(memory_filepath)

    def run(self, task: Task) -> Any:
        raise NotImplementedError("Each agent must implement its own 'run' method for processing a Task object.")
