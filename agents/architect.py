# agents/architect.py
"""
Architect Agent Module

Creates a high-level design before coding begins:
- File structure
- Class/function signatures
- Dependencies between components
- Interface definitions

This gives the Developer agent a blueprint to follow.
"""

import json
import re
from typing import Dict, List

from core.llm_provider import get_llm_client, BaseLLMClient
from core.shared_context import get_shared_context, Architecture, SharedContext


class ArchitectAgent:
    """Agent responsible for creating high-level software architecture designs."""
    
    temperature: float
    client: BaseLLMClient
    shared_context: SharedContext
    
    def __init__(self, temperature: float = 0.2) -> None:
        self.temperature = temperature
        self.client = get_llm_client(temperature=temperature)
        self.shared_context = get_shared_context()

    def design(self, user_prompt: str, tasks: List[str]) -> Architecture:
        """
        Create an architecture design based on user request and planned tasks.
        Returns an Architecture object that will guide development.
        """
        
        tasks_str = "\n".join([f"- {t}" for t in tasks])
        
        system_message = """You are a senior software architect designing a Python application.

Given a project description and planned tasks, create a technical architecture design.

Output your design in this EXACT JSON format:
{
    "description": "Brief overall architecture description",
    "classes": {
        "ClassName": ["attribute1: type", "attribute2: type", "method1(args) -> return_type"]
    },
    "interfaces": {
        "method_name(args) -> return_type": "Description of what this method does"
    },
    "dependencies": {
        "Component1": ["Component2"],
        "Component2": []
    }
}

RULES:
1. Design for simplicity - prefer fewer, well-designed classes
2. Use dataclasses for data models when appropriate
3. Include type hints in signatures
4. Make dependencies clear - what depends on what
5. Output ONLY valid JSON, no markdown, no explanation"""

        user_message = f"""Project: {user_prompt}

Planned Tasks:
{tasks_str}

Design the architecture:"""

        try:
            output = self.client.chat(
                user_message=user_message,
                system_message=system_message,
                temperature=self.temperature,
                max_tokens=1500
            )
            
            # Parse the JSON response
            architecture = self._parse_architecture(output)
            
            # Store in shared context
            self.shared_context.set_architecture(architecture)
            
            return architecture
            
        except Exception as e:
            print(f"  Architect error: {e}")
            return Architecture(description=f"Failed to create architecture: {str(e)}")

    def _parse_architecture(self, output: str) -> Architecture:
        """Parse LLM output into Architecture object."""
        
        # Clean up the output - remove markdown code blocks if present
        output = output.strip()
        if output.startswith("```"):
            # Remove markdown code fence
            lines = output.split("\n")
            # Find start and end of JSON
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            output = "\n".join(lines[start_idx:end_idx])
        
        try:
            data = json.loads(output)
            return Architecture(
                description=data.get("description", ""),
                files=data.get("files", []),
                classes=data.get("classes", {}),
                interfaces=data.get("interfaces", {}),
                dependencies=data.get("dependencies", {})
            )
        except json.JSONDecodeError as e:
            print(f"  Failed to parse architecture JSON: {e}")
            print(f"  Raw output: {output[:500]}...")
            return Architecture(description=output[:500])
    
    def get_design_summary(self) -> str:
        """Get a human-readable summary of the architecture."""
        arch = self.shared_context.architecture
        
        lines = []
        lines.append(f"## Architecture Design")
        lines.append(f"\n{arch.description}")
        
        if arch.classes:
            lines.append("\n### Classes")
            for cls_name, members in arch.classes.items():
                lines.append(f"\n**{cls_name}**:")
                for member in members:
                    lines.append(f"  - {member}")
        
        if arch.interfaces:
            lines.append("\n### Key Interfaces")
            for sig, desc in arch.interfaces.items():
                lines.append(f"  - `{sig}`: {desc}")
        
        if arch.dependencies:
            lines.append("\n### Dependencies")
            for comp, deps in arch.dependencies.items():
                if deps:
                    lines.append(f"  - {comp} depends on: {', '.join(deps)}")
                else:
                    lines.append(f"  - {comp} (no dependencies)")
        
        return "\n".join(lines)
