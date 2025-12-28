"""
Multi-Agent Planner - Agents Package

This package contains all agent implementations:
- PlannerAgent: Breaks tasks into subtasks
- DeveloperAgent: Writes code with sandboxed execution
- QAAgent: Validates code execution
- CriticAgent: Reviews and suggests improvements
"""

__all__ = [
    "PlannerAgent",
    "DeveloperAgent",
    "QAAgent",
    "CriticAgent",
    "BaseAgent",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "PlannerAgent":
        from agents.planner import PlannerAgent
        return PlannerAgent
    elif name == "DeveloperAgent":
        from agents.developer import DeveloperAgent
        return DeveloperAgent
    elif name == "QAAgent":
        from agents.qa import QAAgent
        return QAAgent
    elif name == "CriticAgent":
        from agents.critic import CriticAgent
        return CriticAgent
    elif name == "BaseAgent":
        from agents.base_agent import BaseAgent
        return BaseAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
