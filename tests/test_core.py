"""
Tests for the Multi-Agent Planner system.
"""

import pytest


class TestImports:
    """Test that all modules can be imported."""

    def test_import_agents(self):
        """Test agent imports."""
        from agents import PlannerAgent, DeveloperAgent, QAAgent, CriticAgent
        assert PlannerAgent is not None
        assert DeveloperAgent is not None
        assert QAAgent is not None
        assert CriticAgent is not None

    def test_import_core(self):
        """Test core module imports."""
        from core import (
            get_llm_client,
            execute_code_safely,
            retry_with_backoff,
            Memory,
            Task,
        )
        assert get_llm_client is not None
        assert execute_code_safely is not None
        assert retry_with_backoff is not None
        assert Memory is not None
        assert Task is not None


class TestSandbox:
    """Test sandboxed code execution."""

    def test_safe_code_execution(self):
        """Test that safe code executes correctly."""
        from core.sandbox import execute_code_safely

        result = execute_code_safely('print(1 + 1)')
        assert result["success"] is True
        assert "2" in result["output"]

    def test_dangerous_code_blocked(self):
        """Test that dangerous code is blocked."""
        from core.sandbox import execute_code_safely

        result = execute_code_safely('import os; os.system("ls")')
        assert result["success"] is False
        assert "Security violation" in result.get("error", "")


class TestTaskSchema:
    """Test task schema."""

    def test_task_creation(self):
        """Test task creation."""
        from core.task_schema import Task

        task = Task(id=1, description="Test task")
        assert task.id == 1
        assert task.description == "Test task"
        assert task.status == "pending"
        assert task.result is None


class TestMemory:
    """Test memory system."""

    def test_memory_set_get(self, tmp_path):
        """Test memory set and get operations."""
        from core.memory import Memory

        memory = Memory(filepath=str(tmp_path / "test_memory.json"))
        memory.set("key", "value")
        assert memory.get("key") == "value"
        assert memory.get("nonexistent", "default") == "default"
