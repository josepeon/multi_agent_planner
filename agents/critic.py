# agents/critic.py

import os
import openai
from openai import OpenAIError
from dotenv import load_dotenv
from core.memory import Memory
load_dotenv()

class CriticAgent:
    def __init__(self, model="gpt-4o", temperature=0.3, memory_path="memory/critic_memory.json"):
        self.model = model
        self.temperature = temperature
        self.memory = Memory(memory_path)
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def review(self, task_description, code, error_message):
        system_message = {
            "role": "system",
            "content": (
                "You are a senior code reviewer. Your job is to analyze Python code "
                "and provide constructive feedback. Your feedback should focus on possible causes "
                "of errors, bad practices, or missing edge cases, especially in light of a reported failure."
            )
        }

        user_message = {
            "role": "user",
            "content": (
                f"Task: {task_description}\n\n"
                f"Code:\n{code}\n\n"
                f"Error:\n{error_message}\n\n"
                "What could be improved?"
            )
        }

        cache_key = f"{task_description}|{code}|{error_message}"
        cached_review = self.memory.get(cache_key)
        if cached_review:
            return cached_review

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[system_message, user_message],
                temperature=self.temperature
            )
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content.strip()
                self.memory.set(cache_key, result)
                return result
            else:
                return "No response received from the model."
        except OpenAIError as e:
            return f"OpenAI API error: {str(e)}"