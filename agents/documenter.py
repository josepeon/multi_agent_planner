# agents/documenter.py

"""
Documentation Agent

Generates comprehensive documentation for generated code:
- Adds docstrings to functions/classes
- Generates README.md for the project
- Adds type hints where missing
"""

from core.llm_provider import get_llm_client
from typing import Optional


class DocumenterAgent:
    def __init__(self, temperature=0.2):
        self.temperature = temperature
        self.client = get_llm_client(temperature=temperature)

    def generate_readme(self, code: str, project_description: str) -> str:
        """
        Generate a README.md for the project.
        
        Args:
            code: The source code of the project
            project_description: Original user prompt/description
            
        Returns:
            README.md content as a string
        """
        
        system_message = """You are a technical writer creating documentation.

Generate a professional README.md file with these sections:
1. Project Title and Description
2. Features
3. Installation
4. Usage (with code examples)
5. API Reference (brief overview of classes/functions)
6. Examples

Keep it concise but informative. Use proper Markdown formatting.
Output ONLY the README content - no explanations."""

        user_message = f"""Create a README.md for this project:

**Project Description:** {project_description}

**Source Code:**
```python
{code}
```

Generate the README.md:"""

        try:
            response = self.client.chat(
                user_message=user_message,
                system_message=system_message,
                temperature=self.temperature,
                max_tokens=2000
            )
            return response.strip()
            
        except Exception as e:
            return f"# Project\n\nREADME generation failed: {str(e)}"

    def add_docstrings(self, code: str) -> str:
        """
        Add comprehensive docstrings to code that's missing them.
        
        Args:
            code: Source code to document
            
        Returns:
            Code with added docstrings
        """
        
        system_message = """You are a senior Python developer adding documentation.

RULES:
1. Add Google-style docstrings to ALL functions/methods that don't have them
2. Add class docstrings explaining the purpose
3. Add type hints where missing
4. Keep existing code functionality unchanged
5. Output ONLY valid Python code - no markdown, no explanations

Google-style docstring format:
def func(arg1: int, arg2: str) -> bool:
    '''Brief description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
    '''"""

        user_message = f"""Add docstrings and type hints to this code:

```python
{code}
```

Return the documented code:"""

        try:
            response = self.client.chat(
                user_message=user_message,
                system_message=system_message,
                temperature=self.temperature,
                max_tokens=3000
            )
            return self._clean_code(response)
            
        except Exception as e:
            return code  # Return original if documentation fails

    def _clean_code(self, code: str) -> str:
        """Clean up LLM response."""
        code = code.strip()
        
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
