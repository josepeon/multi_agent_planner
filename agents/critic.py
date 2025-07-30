# agents/critic.py

import os
import openai

class CriticAgent:
    def __init__(self, model="gpt-4", temperature=0.3):
        self.model = model
        self.temperature = temperature
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def critique_code(self, task_description, code, error_message):
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

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[system_message, user_message],
            temperature=self.temperature
        )

        return response["choices"][0]["message"]["content"].strip()
    