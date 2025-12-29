# agents/planner.py
"""
Planner Agent Module

Breaks down user requests into logical development modules using LLM.
"""

import re

from core.llm_provider import get_llm_client
from core.memory import Memory
from core.task_schema import Task

memory = Memory("output/memory.json")


class PlannerAgent:
    """Agent responsible for decomposing user requests into development tasks."""

    temperature: float

    def __init__(self, temperature: float = 0.3) -> None:
        self.temperature = temperature
        self.client = get_llm_client(temperature=temperature)

    def plan_task(self, user_prompt: str) -> list[str]:
        """
        Break down a user prompt into 2-4 logical development modules.

        Args:
            user_prompt: The user's high-level request

        Returns:
            List of module descriptions (2-4 items)
        """
        system_message = """You are a senior software architect planning a development task.

Given a user's request, break it into 2-4 LOGICAL MODULES (not micro-tasks).

IMPORTANT RULES:
1. Create 2-4 modules maximum, each representing a COMPLETE, TESTABLE component
2. Each module should produce runnable Python code that can be tested independently
3. AVOID splitting into tiny tasks like "add a method" - group related functionality
4. Think in terms of: Data Models, Core Logic, Interface/API, Main Entry Point

GOOD EXAMPLE for "Build a todo list manager":
1. Define Task data class with id, title, completed fields and TaskStatus enum
2. Implement TaskManager class with add_task, complete_task, list_tasks methods
3. Create main() function with example usage demonstrating all features

BAD EXAMPLE (too granular):
1. Create Task class
2. Add title attribute
3. Add completed attribute
4. Create TaskManager
5. Implement add method
6. Implement remove method
7. Implement list method
8. Create main function

Output ONLY a numbered list of 2-4 modules. Each module description should be complete
enough that a developer can implement it without guessing."""

        user_message = f"User request: {user_prompt}"

        try:
            output = self.client.chat(
                user_message=user_message,
                system_message=system_message,
                temperature=self.temperature
            )
        except Exception as e:
            return [f"LLM API error: {str(e)}"]

        subtasks: list[str] = [
            re.sub(r"^\s*\d+[\.\):\-]\s*", "", line).strip()
            for line in output.split("\n")
            if re.match(r"^\s*\d+[\.\):\-]", line)
        ]

        # Limit to 4 tasks max to prevent over-fragmentation
        if len(subtasks) > 4:
            subtasks = subtasks[:4]

        return subtasks

    def plan(self, user_prompt: str) -> list[Task]:
        """
        Plan tasks for a user prompt, with caching support.

        Args:
            user_prompt: The user's high-level request

        Returns:
            List of Task objects representing the development plan
        """
        cache_key = f"plan::{user_prompt}"
        cached = memory.get(cache_key)
        if cached:
            return [Task(**t) if isinstance(t, dict) else t for t in cached]

        result = self.plan_task(user_prompt)
        tasks = [Task(id=i, description=desc) for i, desc in enumerate(result)]
        memory.set(cache_key, tasks)
        return tasks
