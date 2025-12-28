"""
Tests for the Multi-Agent Planner system.
"""

import pytest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock


# ===========================================
# Import Tests
# ===========================================

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

    def test_import_integrator(self):
        """Test integrator agent import."""
        with patch('core.llm_provider.get_llm_client'):
            from agents.integrator import IntegratorAgent
            assert IntegratorAgent is not None

    def test_import_architect(self):
        """Test architect agent import."""
        with patch('core.llm_provider.get_llm_client'):
            from agents.architect import ArchitectAgent
            assert ArchitectAgent is not None

    def test_import_test_generator(self):
        """Test test generator agent import."""
        with patch('core.llm_provider.get_llm_client'):
            from agents.test_generator import TestGeneratorAgent
            assert TestGeneratorAgent is not None

    def test_import_documenter(self):
        """Test documenter agent import."""
        with patch('core.llm_provider.get_llm_client'):
            from agents.documenter import DocumenterAgent
            assert DocumenterAgent is not None


# ===========================================
# Sandbox Tests
# ===========================================

class TestSandbox:
    """Test sandboxed code execution."""

    def test_safe_code_execution(self):
        """Test that safe code executes correctly."""
        from core.sandbox import execute_code_safely

        result = execute_code_safely('print(1 + 1)')
        assert result["success"] is True
        assert "2" in result["output"]

    def test_math_operations(self):
        """Test math module works in sandbox."""
        from core.sandbox import execute_code_safely

        result = execute_code_safely('import math; print(math.sqrt(16))')
        assert result["success"] is True
        assert "4" in result["output"]

    def test_class_definition(self):
        """Test that class definitions work in sandbox."""
        from core.sandbox import execute_code_safely

        code = '''
class Calculator:
    def add(self, a, b):
        return a + b

calc = Calculator()
print(calc.add(2, 3))
'''
        result = execute_code_safely(code)
        assert result["success"] is True
        assert "5" in result["output"]

    def test_dataclass_support(self):
        """Test that dataclasses work in sandbox."""
        from core.sandbox import execute_code_safely

        code = '''
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

p = Person("Alice", 30)
print(p.name, p.age)
'''
        result = execute_code_safely(code)
        assert result["success"] is True
        assert "Alice" in result["output"]
        assert "30" in result["output"]

    def test_enum_support(self):
        """Test that enums work in sandbox."""
        from core.sandbox import execute_code_safely

        code = '''
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    DONE = "done"

print(Status.PENDING.value)
'''
        result = execute_code_safely(code)
        assert result["success"] is True
        assert "pending" in result["output"]

    def test_dangerous_os_system_blocked(self):
        """Test that os.system is blocked."""
        from core.sandbox import execute_code_safely

        result = execute_code_safely('import os; os.system("ls")')
        assert result["success"] is False
        assert "Security violation" in result.get("error", "")

    def test_dangerous_subprocess_blocked(self):
        """Test that subprocess is blocked."""
        from core.sandbox import execute_code_safely

        result = execute_code_safely('import subprocess; subprocess.run(["ls"])')
        assert result["success"] is False
        assert "Security violation" in result.get("error", "")

    def test_dangerous_popen_blocked(self):
        """Test that Popen is blocked."""
        from core.sandbox import execute_code_safely

        result = execute_code_safely('from subprocess import Popen')
        assert result["success"] is False
        assert "Security violation" in result.get("error", "")

    def test_disallowed_import_blocked(self):
        """Test that non-whitelisted imports are blocked."""
        from core.sandbox import execute_code_safely

        result = execute_code_safely('import socket')
        assert result["success"] is False
        assert "not allowed" in result.get("error", "").lower()

    def test_list_comprehension(self):
        """Test list comprehensions work."""
        from core.sandbox import execute_code_safely

        result = execute_code_safely('print([x**2 for x in range(5)])')
        assert result["success"] is True
        assert "[0, 1, 4, 9, 16]" in result["output"]

    def test_exception_handling(self):
        """Test exception handling in sandbox."""
        from core.sandbox import execute_code_safely

        code = '''
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Caught division by zero")
'''
        result = execute_code_safely(code)
        assert result["success"] is True
        assert "Caught division by zero" in result["output"]

    def test_json_operations(self):
        """Test JSON module works."""
        from core.sandbox import execute_code_safely

        code = '''
import json
data = {"name": "test", "value": 42}
print(json.dumps(data))
'''
        result = execute_code_safely(code)
        assert result["success"] is True
        assert '"name": "test"' in result["output"] or '"name":"test"' in result["output"]


# ===========================================
# Task Schema Tests
# ===========================================

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

    def test_task_with_code(self):
        """Test task with code result."""
        from core.task_schema import Task

        task = Task(id=1, description="Test", result="print('hello')")
        assert task.result == "print('hello')"

    def test_task_status_update(self):
        """Test task status can be updated."""
        from core.task_schema import Task

        task = Task(id=1, description="Test")
        task.status = "completed"
        assert task.status == "completed"


# ===========================================
# Memory Tests
# ===========================================

class TestMemory:
    """Test memory system."""

    def test_memory_set_get(self, tmp_path):
        """Test memory set and get operations."""
        from core.memory import Memory

        memory = Memory(filepath=str(tmp_path / "test_memory.json"))
        memory.set("key", "value")
        assert memory.get("key") == "value"
        assert memory.get("nonexistent", "default") == "default"

    def test_memory_persistence(self, tmp_path):
        """Test memory persists to file."""
        from core.memory import Memory

        filepath = str(tmp_path / "test_memory.json")
        
        # Create and set
        memory1 = Memory(filepath=filepath)
        memory1.set("persistent_key", "persistent_value")
        
        # Load new instance
        memory2 = Memory(filepath=filepath)
        assert memory2.get("persistent_key") == "persistent_value"

    def test_memory_complex_values(self, tmp_path):
        """Test memory with complex data structures."""
        from core.memory import Memory

        memory = Memory(filepath=str(tmp_path / "test_memory.json"))
        complex_data = {
            "list": [1, 2, 3],
            "nested": {"a": "b"},
            "number": 42
        }
        memory.set("complex", complex_data)
        retrieved = memory.get("complex")
        assert retrieved == complex_data

    def test_memory_overwrite(self, tmp_path):
        """Test memory overwrites existing keys."""
        from core.memory import Memory

        memory = Memory(filepath=str(tmp_path / "test_memory.json"))
        memory.set("key", "value1")
        memory.set("key", "value2")
        assert memory.get("key") == "value2"


# ===========================================
# Retry Logic Tests
# ===========================================

class TestRetryLogic:
    """Test retry mechanism."""

    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        from core.retry import RetryConfig

        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.jitter is True

    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation."""
        from core.retry import calculate_delay, RetryConfig

        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        assert calculate_delay(0, config) == 1.0
        assert calculate_delay(1, config) == 2.0
        assert calculate_delay(2, config) == 4.0

    def test_calculate_delay_max_cap(self):
        """Test delay is capped at max_delay."""
        from core.retry import calculate_delay, RetryConfig

        config = RetryConfig(base_delay=10.0, max_delay=20.0, jitter=False)
        
        # 10 * 2^3 = 80, should be capped at 20
        assert calculate_delay(3, config) == 20.0

    def test_retry_decorator_success(self):
        """Test retry decorator with immediate success."""
        from core.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def succeed_immediately():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed_immediately()
        assert result == "success"
        assert call_count == 1

    def test_retry_decorator_eventual_success(self):
        """Test retry decorator with eventual success."""
        from core.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 3


# ===========================================
# LLM Config Tests
# ===========================================

class TestLLMConfig:
    """Test LLM configuration."""

    def test_default_config(self):
        """Test default LLM configuration."""
        from core.llm_provider import LLMConfig

        config = LLMConfig()
        assert config.provider == "groq"
        assert config.model == "llama-3.3-70b-versatile"
        assert config.temperature == 0.3

    def test_custom_provider(self):
        """Test custom provider configuration."""
        from core.llm_provider import LLMConfig

        config = LLMConfig(provider="gemini")
        assert config.provider == "gemini"
        assert config.model == "gemini-2.0-flash-exp"

    def test_custom_model(self):
        """Test custom model configuration."""
        from core.llm_provider import LLMConfig

        config = LLMConfig(provider="groq", model="custom-model")
        assert config.model == "custom-model"


# ===========================================
# Groq Client Tests (Mocked)
# ===========================================

class TestGroqClientMocked:
    """Test Groq client with mocked API."""

    def test_groq_client_chat(self):
        """Test Groq client chat method."""
        from core.llm_provider import GroqClient, LLMConfig
        
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, World!"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Patch Groq where it's imported
        with patch('groq.Groq', return_value=mock_client):
            with patch.dict('os.environ', {'GROQ_API_KEY': 'test-key'}):
                config = LLMConfig(provider="groq")
                client = GroqClient(config)
                result = client.chat("Say hello")
                
                assert result == "Hello, World!"
                mock_client.chat.completions.create.assert_called_once()

    def test_groq_client_fallback_on_rate_limit(self):
        """Test Groq client falls back to different model on rate limit."""
        from core.llm_provider import GroqClient, LLMConfig
        
        # Setup mock
        mock_client = MagicMock()
        
        # First call raises rate limit, second succeeds
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Fallback success"
        
        mock_client.chat.completions.create.side_effect = [
            Exception("rate_limit exceeded"),
            mock_response
        ]
        
        with patch('groq.Groq', return_value=mock_client):
            with patch.dict('os.environ', {'GROQ_API_KEY': 'test-key'}):
                config = LLMConfig(provider="groq")
                client = GroqClient(config)
                result = client.chat("Test")
                
                assert result == "Fallback success"
                assert len(mock_client.chat.completions.create.call_args_list) == 2

    def test_groq_fallback_models_defined(self):
        """Test Groq client has fallback models defined."""
        from core.llm_provider import GroqClient
        
        assert hasattr(GroqClient, 'FALLBACK_MODELS')
        assert len(GroqClient.FALLBACK_MODELS) >= 1


# ===========================================
# Shared Context Tests
# ===========================================

class TestSharedContext:
    """Test shared context system."""

    def test_shared_context_creation(self, tmp_path):
        """Test shared context initialization."""
        from core.shared_context import SharedContext

        ctx = SharedContext(filepath=str(tmp_path / "context.json"))
        assert ctx is not None

    def test_add_generated_code(self, tmp_path):
        """Test adding generated code to shared context."""
        from core.shared_context import SharedContext

        ctx = SharedContext(filepath=str(tmp_path / "context.json"))
        ctx.add_generated_code(
            task_id=1,
            name="TestClass",
            code="class TestClass:\n    pass",
            status="passed"
        )
        
        summary = ctx.get_context_summary()
        assert "TestClass" in summary

    def test_add_class_to_context(self, tmp_path):
        """Test adding class to defined classes."""
        from core.shared_context import SharedContext

        ctx = SharedContext(filepath=str(tmp_path / "context.json"))
        ctx.defined_classes["Calculator"] = "class Calculator:\n    def add(self, a, b): return a + b"
        
        assert "Calculator" in ctx.defined_classes

    def test_context_persistence(self, tmp_path):
        """Test context is saved and loaded."""
        from core.shared_context import SharedContext

        filepath = str(tmp_path / "context.json")
        
        # Create and save
        ctx1 = SharedContext(filepath=filepath)
        ctx1.imports.add("json")
        ctx1._save()
        
        # Load in new instance
        ctx2 = SharedContext(filepath=filepath)
        assert "json" in ctx2.imports


# ===========================================
# Base Agent Tests
# ===========================================

class TestBaseAgent:
    """Test base agent functionality."""

    def test_base_agent_creation(self, tmp_path):
        """Test base agent initialization."""
        from agents.base_agent import BaseAgent

        agent = BaseAgent("test_agent", str(tmp_path / "memory.json"))
        assert agent.name == "test_agent"
        assert agent.memory is not None

    def test_base_agent_run_not_implemented(self, tmp_path):
        """Test base agent run raises NotImplementedError."""
        from agents.base_agent import BaseAgent
        from core.task_schema import Task

        agent = BaseAgent("test_agent")
        task = Task(id=1, description="Test")
        
        with pytest.raises(NotImplementedError):
            agent.run(task)


# ===========================================
# Integration Tests
# ===========================================

class TestIntegration:
    """Integration tests for the system."""

    def test_sandbox_with_generated_code(self):
        """Test sandbox executes typical generated code patterns."""
        from core.sandbox import execute_code_safely

        # Typical generated code pattern
        code = '''
from dataclasses import dataclass
from enum import Enum
from typing import List

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class Todo:
    title: str
    priority: Priority
    done: bool = False

todos: List[Todo] = []
todos.append(Todo("Test task", Priority.HIGH))
print(f"Created {len(todos)} todo(s)")
print(f"First todo: {todos[0].title}")
'''
        result = execute_code_safely(code)
        assert result["success"] is True
        assert "Created 1 todo" in result["output"]
        assert "Test task" in result["output"]

    def test_memory_with_agent_pattern(self, tmp_path):
        """Test memory works with typical agent usage pattern."""
        from core.memory import Memory

        memory = Memory(filepath=str(tmp_path / "agent_memory.json"))
        
        # Simulate agent caching LLM response
        prompt_hash = "abc123"
        response = "def add(a, b):\n    return a + b"
        
        memory.set(f"llm_response_{prompt_hash}", response)
        
        # Retrieve cached response
        cached = memory.get(f"llm_response_{prompt_hash}")
        assert cached == response
