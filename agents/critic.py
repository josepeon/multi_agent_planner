# agents/critic.py
"""
Critic Agent Module

Reviews code and provides constructive feedback for failed execution attempts.
"""


from core.llm_provider import BaseLLMClient, get_llm_client
from core.memory import Memory


class CriticAgent:
    """Agent responsible for reviewing code and providing improvement suggestions."""

    temperature: float
    client: BaseLLMClient
    memory: Memory

    def __init__(
        self,
        temperature: float = 0.3,
        memory_path: str = "memory/critic_memory.json"
    ) -> None:
        self.temperature = temperature
        self.client = get_llm_client(temperature=temperature, max_tokens=1024)
        self.memory = Memory(memory_path)

    def review(self, task_description: str, code: str, error_message: str) -> str:
        """
        Review code that failed execution and provide improvement suggestions.
        
        Args:
            task_description: What the code was supposed to accomplish
            code: The Python code that failed
            error_message: Error or output from execution
            
        Returns:
            Review feedback as string
        """
        system_message = (
            "You are a senior code reviewer. Your job is to analyze Python code "
            "and provide constructive feedback. Your feedback should focus on possible causes "
            "of errors, bad practices, or missing edge cases, especially in light of a reported failure."
        )

        user_message = (
            f"Task: {task_description}\n\n"
            f"Code:\n{code}\n\n"
            f"Error:\n{error_message}\n\n"
            "What could be improved?"
        )

        cache_key = f"{task_description}|{code}|{error_message}"
        cached_review: str | None = self.memory.get(cache_key)
        if cached_review:
            return cached_review

        try:
            result: str = self.client.chat(
                user_message=user_message,
                system_message=system_message,
                temperature=self.temperature
            )
            self.memory.set(cache_key, result)
            return result
        except Exception as e:
            return f"LLM API error: {str(e)}"
