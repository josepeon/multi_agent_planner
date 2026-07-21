# agents/test_generator.py
"""
Test Generator Agent Module

Generates pytest unit tests for generated code.
Analyzes classes and functions to create comprehensive test coverage.
"""


from core.languages import LanguageProfile, get_profile
from core.llm_provider import BaseLLMClient, get_llm_client
from core.shared_context import SharedContext, get_shared_context


class TestGeneratorAgent:
    """Agent responsible for generating pytest unit tests."""

    temperature: float
    client: BaseLLMClient
    shared_context: SharedContext

    def __init__(
        self,
        temperature: float = 0.2,
        language: LanguageProfile | None = None,
    ) -> None:
        self.temperature = temperature
        self.language = language or get_profile()
        self.client = get_llm_client(temperature=temperature, role="test_generator")
        self.shared_context = get_shared_context()

    def generate_tests(self, code: str, project_name: str = "project") -> str:
        """
        Generate pytest unit tests for the given code.

        Args:
            code: The source code to generate tests for
            project_name: Name of the project (used for file naming)

        Returns:
            pytest test code as a string
        """

        system_message = """You are a senior QA engineer writing pytest unit tests.

RULES:
1. Generate comprehensive pytest tests for ALL classes and functions
2. Include edge cases, error handling, and boundary conditions
3. Use pytest fixtures where appropriate
4. Use descriptive test names: test_<function>_<scenario>
5. Include docstrings explaining each test
6. Test both positive and negative cases
7. Output ONLY valid Python pytest code - no explanations or markdown

Example format:
```python
import pytest
from dataclasses import dataclass

# Test fixtures
@pytest.fixture
def sample_instance():
    return MyClass()

# Test cases
class TestMyClass:
    def test_method_success(self, sample_instance):
        '''Test method with valid input'''
        result = sample_instance.method(valid_input)
        assert result == expected

    def test_method_raises_on_invalid(self, sample_instance):
        '''Test method raises ValueError on invalid input'''
        with pytest.raises(ValueError):
            sample_instance.method(invalid_input)
```"""

        user_message = f"""Generate pytest unit tests for this code:

```python
{code}
```

Generate comprehensive tests covering:
1. Normal operation
2. Edge cases (empty, single item, large values)
3. Error handling (invalid inputs, boundary conditions)
4. Property/method behavior

Output ONLY the pytest code:"""

        try:
            response = self.client.chat(
                user_message=user_message,
                system_message=system_message,
                temperature=self.temperature,
                max_tokens=2500
            )

            test_code = self._clean_code(response)
            return test_code

        except Exception as e:
            return f"# Test generation failed: {str(e)}"

    def _clean_code(self, code: str) -> str:
        """Clean up LLM response to extract pure Python code."""
        from agents.base_agent import strip_code_fences
        return strip_code_fences(code)
