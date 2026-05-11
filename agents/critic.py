# agents/critic.py
"""
Critic Agent Module

Reviews code and provides constructive feedback for failed execution attempts.
"""


from core.cache import BoundedCache
from core.llm_provider import BaseLLMClient, get_llm_client


class CriticAgent:
    """Agent responsible for reviewing code and providing improvement suggestions."""

    temperature: float
    client: BaseLLMClient
    memory: BoundedCache

    def __init__(
        self,
        temperature: float = 0.3,
        memory_path: str = "memory/critic_memory.json"
    ) -> None:
        self.temperature = temperature
        self.client = get_llm_client(temperature=temperature, max_tokens=1024)
        # Bounded LRU cache for review memoization. Same defaults as developer:
        # cap = 1000 entries, TTL = 30 days. Loads legacy unbounded JSON cleanly.
        self.memory = BoundedCache(
            memory_path,
            max_size=1000,
            ttl_seconds=60 * 60 * 24 * 30,
        )

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

    def score(self, task_description: str, code: str) -> float:
        """Score code 0-10 on a best-of-N generation. Returns a float.

        Used by core.orchestrator when BEST_OF_N>1: each candidate gets a
        score, the highest-scoring one wins. The critic is asked to be
        terse — we only need a number.
        """
        system_message = (
            "You are a senior code reviewer. Score the following Python code "
            "on a scale of 0-10 (10 = excellent) for correctness, clarity, "
            "and idiomatic style relative to the task. Respond with ONLY the "
            "number, nothing else."
        )
        user_message = (
            f"Task: {task_description}\n\n"
            f"Code:\n{code}\n\n"
            f"Score (0-10):"
        )
        try:
            raw = self.client.chat(
                user_message=user_message,
                system_message=system_message,
                temperature=0.1,
                max_tokens=8,
            )
        except Exception:
            return 0.0

        # Parse the first number out of the response; clamp to [0, 10].
        import re

        match = re.search(r"\d+(?:\.\d+)?", raw)
        if not match:
            return 0.0
        try:
            value = float(match.group())
        except ValueError:
            return 0.0
        return max(0.0, min(10.0, value))
