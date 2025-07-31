# agents/developer.py

import os
import openai
from openai import OpenAIError
from dotenv import load_dotenv
load_dotenv()

class DeveloperAgent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def write_code(self, task_description, feedback_message=None):
        system_message = {
            "role": "system",
            "content": (
                "You are a senior Python developer. "
                "Your job is to write a clean, minimal Python function or code block "
                "that fulfills a single, clearly defined task. "
                "Only return valid Python code — no explanations or markdown."
            )
        }
        base_prompt = f"Task: {task_description}"
        if feedback_message:
            base_prompt += f"\nNote: Your previous attempt failed. Error was:\n{feedback_message}"

        user_message = {
            "role": "user",
            "content": base_prompt
        }

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[system_message, user_message],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            return f"OpenAI API error: {str(e)}"