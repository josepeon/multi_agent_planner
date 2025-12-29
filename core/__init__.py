"""
Multi-Agent Planner - Core Package

This package contains core functionality:
- orchestrator: Pipeline coordination
- llm_provider: Multi-provider LLM abstraction
- sandbox: Sandboxed code execution
- retry: Exponential backoff retry logic
- memory: Persistent JSON memory
- task_schema: Task dataclass
- logger: Structured logging
"""

__all__ = [
    # Orchestrator
    "run_orchestrator",
    "run_pipeline",
    # LLM
    "get_llm_client",
    "LLMConfig",
    # Sandbox
    "execute_code_safely",
    # Retry
    "retry_with_backoff",
    "retry_llm_call",
    # Memory
    "Memory",
    # Schema
    "Task",
    # Logging
    "get_logger",
    "setup_logging",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in ("run_orchestrator", "run_pipeline"):
        from core.orchestrator import run_orchestrator, run_pipeline
        return run_orchestrator if name == "run_orchestrator" else run_pipeline
    elif name in ("get_llm_client", "LLMConfig"):
        from core.llm_provider import LLMConfig, get_llm_client
        return get_llm_client if name == "get_llm_client" else LLMConfig
    elif name == "execute_code_safely":
        from core.sandbox import execute_code_safely
        return execute_code_safely
    elif name in ("retry_with_backoff", "retry_llm_call"):
        from core.retry import retry_llm_call, retry_with_backoff
        return retry_with_backoff if name == "retry_with_backoff" else retry_llm_call
    elif name == "Memory":
        from core.memory import Memory
        return Memory
    elif name == "Task":
        from core.task_schema import Task
        return Task
    elif name in ("get_logger", "setup_logging"):
        from core.logger import get_logger, setup_logging
        return get_logger if name == "get_logger" else setup_logging
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
