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

from agents.base_agent import extract_json
from core.llm_provider import BaseLLMClient, get_llm_client
from core.memory_store import UserMemory
from core.shared_context import Architecture, SharedContext, get_shared_context


class ArchitectAgent:
    """Agent responsible for creating high-level software architecture designs."""

    temperature: float
    client: BaseLLMClient
    shared_context: SharedContext
    user_memory: UserMemory | None

    def __init__(
        self,
        temperature: float = 0.2,
        user_memory: UserMemory | None = None,
    ) -> None:
        self.temperature = temperature
        self.client = get_llm_client(temperature=temperature, role="architect")
        self.shared_context = get_shared_context()
        # Lazy: if the caller didn't supply one, instantiate a default-backed
        # UserMemory. The default backend is JSON-on-disk; no external deps.
        self.user_memory = user_memory if user_memory is not None else UserMemory()

    def design(
        self,
        user_prompt: str,
        tasks: list[str],
        research_brief: str = "",
    ) -> Architecture:
        """
        Create an architecture design based on user request and planned tasks.

        Optional ``research_brief`` is text from the Researcher agent; if
        supplied it's included in the architect's prompt so design decisions
        can reflect current library APIs and best practices.

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

        # Pull any recorded user preferences relevant to this prompt. Empty
        # string if no UserMemory was supplied or no entries matched.
        preferences_block = ""
        if self.user_memory is not None:
            remembered = self.user_memory.render(
                query=user_prompt + " " + tasks_str,
                k=5,
            )
            if remembered:
                preferences_block = (
                    "\n\nRemembered preferences and lessons from prior runs "
                    "— honor these unless they conflict with the current "
                    "request:\n" + remembered + "\n"
                )

        research_block = f"\n\n{research_brief}\n" if research_brief else ""

        user_message = f"""Project: {user_prompt}

Planned Tasks:
{tasks_str}{preferences_block}{research_block}

Design the architecture:"""

        try:
            output = self.client.chat(
                user_message=user_message,
                system_message=system_message,
                temperature=self.temperature,
                max_tokens=1500,
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
        """Parse LLM output into Architecture object.

        Uses the shared extract_json helper, which tolerates fences and
        leading/trailing prose — a reply of valid JSON plus a closing
        sentence used to silently collapse the whole design to a text blob.
        """
        data = extract_json(output)
        if isinstance(data, dict):
            return Architecture(
                description=data.get("description", ""),
                files=data.get("files", []),
                classes=data.get("classes", {}),
                interfaces=data.get("interfaces", {}),
                dependencies=data.get("dependencies", {}),
            )
        print("  Failed to parse architecture JSON")
        print(f"  Raw output: {output[:500]}...")
        return Architecture(description=output[:500])

    def get_design_summary(self) -> str:
        """Get a human-readable summary of the architecture."""
        arch = self.shared_context.architecture

        lines = []
        lines.append("## Architecture Design")
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
