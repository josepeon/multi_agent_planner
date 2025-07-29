# agents/developer.py

import os
import openai

class DeveloperAgent:
    def __init__(self, model="gpt-4"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def write_code(self, task_description):
        system_message = {
            "role": "system",
            "content": (
                "You are a senior Python developer. "
                "Your job is to write a clean, minimal Python function or code block "
                "that fulfills a single, clearly defined task. "
                "Only return valid Python code — no explanations or markdown."
            )
        }

        user_message = {
            "role": "user",
            "content": f"Task: {task_description}"
        }

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[system_message, user_message],
            temperature=0.3
        )

        code = response["choices"][0]["message"]["content"]
        return code.strip()