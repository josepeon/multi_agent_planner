# agents/critic.py

from core.llm_provider import get_llm_client
from core.memory import Memory
from core.task_schema import Task


class CriticAgent:
    def __init__(self, temperature=0.3, memory_path="memory/critic_memory.json"):
        self.temperature = temperature
        self.client = get_llm_client(temperature=temperature, max_tokens=1024)
        self.memory = Memory(memory_path)

    def review(self, task_description: str, code: str, error_message: str):
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
        cached_review = self.memory.get(cache_key)
        if cached_review:
            return cached_review

        try:
            result = self.client.chat(
                user_message=user_message,
                system_message=system_message,
                temperature=self.temperature
            )
            self.memory.set(cache_key, result)
            return result
        except Exception as e:
            return f"LLM API error: {str(e)}"