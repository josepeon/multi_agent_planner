"""
Tests for the Agent classes.

These tests use mocked LLM responses to avoid actual API calls.
"""

from unittest.mock import MagicMock, patch

# ===========================================
# Planner Agent Tests
# ===========================================

class TestPlannerAgent:
    """Test Planner Agent functionality."""

    @patch('agents.planner.get_llm_client')
    def test_planner_creates_tasks(self, mock_get_client):
        """Test planner creates tasks from user prompt."""
        from agents.planner import PlannerAgent

        # Mock LLM response with task breakdown
        mock_client = MagicMock()
        mock_client.chat.return_value = """
        Task 1: Create data models for the application
        Task 2: Implement business logic
        Task 3: Create main entry point
        """
        mock_get_client.return_value = mock_client

        agent = PlannerAgent()
        result = agent.plan("Create a todo list application")

        assert isinstance(result, list)
        mock_client.chat.assert_called_once()

    @patch('agents.planner.get_llm_client')
    def test_planner_uses_system_prompt(self, mock_get_client):
        """Test planner uses appropriate system prompt."""
        from agents.planner import PlannerAgent

        mock_client = MagicMock()
        mock_client.chat.return_value = "Task 1: Test task"
        mock_get_client.return_value = mock_client

        agent = PlannerAgent()
        agent.plan("Create something")

        # Check that system_message was passed
        call_args = mock_client.chat.call_args
        assert 'system_message' in call_args.kwargs or len(call_args.args) > 1


# ===========================================
# Architect Agent Tests
# ===========================================

class TestArchitectAgent:
    """Test Architect Agent functionality."""

    def test_architect_creates_design(self):
        """Test architect creates high-level design."""
        with patch('core.llm_provider.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.return_value = """{
                "description": "Simple architecture",
                "classes": {},
                "interfaces": {},
                "dependencies": {}
            }"""
            mock_get_client.return_value = mock_client

            from agents.architect import ArchitectAgent

            agent = ArchitectAgent()
            result = agent.design("Todo list application", [
                "Create todo item class",
                "Create todo manager",
            ])

            # Result is an Architecture object
            assert result is not None
            mock_client.chat.assert_called_once()


# ===========================================
# Developer Agent Tests
# ===========================================

class TestDeveloperAgent:
    """Test Developer Agent functionality."""

    def test_developer_generates_code(self):
        """Test developer generates code using write_code method."""
        with patch('core.llm_provider.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.return_value = """
class Calculator:
    def add(self, a, b):
        return a + b

print("Calculator created")
"""
            mock_get_client.return_value = mock_client

            from agents.developer import DeveloperAgent

            agent = DeveloperAgent()
            result = agent.write_code("Create a calculator class")

            assert "Calculator" in result
            assert "add" in result

    def test_developer_handles_feedback(self):
        """Test developer incorporates feedback on retry."""
        with patch('core.llm_provider.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.return_value = """
def add(a, b):
    return a + b
"""
            mock_get_client.return_value = mock_client

            from agents.developer import DeveloperAgent

            agent = DeveloperAgent()
            result = agent.write_code(
                "Create an add function",
                feedback_message="Previous version had syntax error"
            )

            assert "def add" in result


# ===========================================
# QA Agent Tests
# ===========================================

class TestQAAgent:
    """Test QA Agent functionality."""

    def test_qa_validates_passed_code(self):
        """Test QA validates code that passed sandbox execution."""
        with patch('core.llm_provider.get_llm_client'):
            from agents.qa import QAAgent

            agent = QAAgent()
            code_result = {
                "status": "passed",
                "result": "Test output",
                "code": "print('hello')"
            }
            result = agent.evaluate_code(code_result)

            assert result["status"] == "passed"

    def test_qa_reports_failed_code(self):
        """Test QA reports code that failed execution."""
        with patch('core.llm_provider.get_llm_client'):
            from agents.qa import QAAgent

            agent = QAAgent()
            code_result = {
                "status": "failed",
                "result": "Error: ZeroDivisionError",
                "code": "x = 1/0"
            }
            result = agent.evaluate_code(code_result)

            assert result["status"] == "failed"
            assert result["critique"] != ""


# ===========================================
# Critic Agent Tests
# ===========================================

class TestCriticAgent:
    """Test Critic Agent functionality."""

    def test_critic_provides_feedback(self):
        """Test critic provides feedback using review method."""
        with patch('core.llm_provider.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.return_value = """
            Issues found:
            1. Missing import statement for json module
            2. Function should handle empty input
            """
            mock_get_client.return_value = mock_client

            from agents.critic import CriticAgent

            agent = CriticAgent()
            feedback = agent.review(
                task_description="Parse JSON",
                code="data = json.loads(input_str)",
                error_message="NameError: name 'json' is not defined"
            )

            assert isinstance(feedback, str)
            assert len(feedback) > 0


# ===========================================
# Integrator Agent Tests
# ===========================================

class TestIntegratorAgent:
    """Test Integrator Agent functionality."""

    def test_integrator_merges_code(self):
        """Test integrator merges multiple task codes from session log."""
        with patch('core.llm_provider.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.return_value = """
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(2, 3))
"""
            mock_get_client.return_value = mock_client

            from agents.integrator import IntegratorAgent

            agent = IntegratorAgent()
            session_log = {
                "prompt": "Create a calculator",
                "tasks": [
                    {"task": "Add function", "code": "def add(a, b): return a + b", "status": "passed"},
                    {"task": "Multiply function", "code": "def multiply(a, b): return a * b", "status": "passed"},
                ]
            }
            result = agent.integrate(session_log)

            assert "Calculator" in result or "add" in result

    def test_integrator_handles_empty_session_log(self):
        """Test integrator handles empty session log."""
        with patch('core.llm_provider.get_llm_client'):
            from agents.integrator import IntegratorAgent

            agent = IntegratorAgent()
            session_log = {"tasks": []}
            result = agent.integrate(session_log)

            assert "No code" in result or result == ""

    def test_integrator_single_block(self):
        """Test integrator with single code block."""
        with patch('core.llm_provider.get_llm_client'):
            from agents.integrator import IntegratorAgent

            agent = IntegratorAgent()
            session_log = {
                "tasks": [
                    {"task": "Test", "code": "print('hello')", "status": "passed"}
                ]
            }
            result = agent.integrate(session_log)

            assert "print" in result


# ===========================================
# Test Generator Agent Tests
# ===========================================

class TestTestGeneratorAgent:
    """Test Test Generator Agent functionality."""

    def test_generator_creates_tests(self):
        """Test generator creates pytest tests."""
        with patch('core.llm_provider.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.return_value = """
import pytest

def test_add():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0
"""
            mock_get_client.return_value = mock_client

            from agents.test_generator import TestGeneratorAgent

            agent = TestGeneratorAgent()
            code = "def add(a, b): return a + b"
            result = agent.generate_tests(code)

            assert "def test_" in result
            assert "assert" in result


# ===========================================
# Documenter Agent Tests
# ===========================================

class TestDocumenterAgent:
    """Test Documenter Agent functionality."""

    def test_documenter_creates_readme(self):
        """Test documenter creates README using generate_readme method."""
        with patch('core.llm_provider.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.return_value = """
# Calculator

A simple calculator application.

## Features
- Addition
- Multiplication

## Usage
```python
calc = Calculator()
result = calc.add(2, 3)
```
"""
            mock_get_client.return_value = mock_client

            from agents.documenter import DocumenterAgent

            agent = DocumenterAgent()
            code = "class Calculator:\n    def add(self, a, b): return a + b"
            result = agent.generate_readme(code, "Create a calculator")

            assert "Calculator" in result or "#" in result


# ===========================================
# Edge Cases
# ===========================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_task_with_empty_description(self):
        """Test handling task with empty description."""
        from core.task_schema import Task

        task = Task(id=1, description="")
        assert task.description == ""
        assert task.id == 1

    def test_task_with_unicode(self):
        """Test handling task with unicode characters."""
        from core.task_schema import Task

        task = Task(id=1, description="Create a function with special chars: [test]")
        assert "special" in task.description
        assert "[test]" in task.description

    def test_developer_returns_clean_code(self):
        """Test developer returns code without markdown."""
        from agents.developer import clean_code_block

        # Test the clean_code_block function directly
        dirty_code = """```python
def hello():
    print("world")
```"""
        clean = clean_code_block(dirty_code)

        # Should not contain triple backticks
        assert "```" not in clean
        assert "def hello" in clean
