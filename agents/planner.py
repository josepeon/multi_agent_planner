# agents/planner.py

import os
import openai

class PlannerAgent:
    def __init__(self, model="gpt-4"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def plan_task(self, user_prompt):
        system_message = {
            "role": "system",
            "content": (
                "You are a planning assistant. Given a user's request, "
                "break the task down into a numbered list of clear, atomic subtasks. "
                "Keep the list focused and executable by a developer agent."
            )
        }

        user_message = {
            "role": "user",
            "content": f"User request: {user_prompt}"
        }

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[system_message, user_message],
            temperature=0.3
        )

        output = response["choices"][0]["message"]["content"]
        # Parse the subtasks (naively assuming numbered list)
        subtasks = [line.strip()[3:].strip() for line in output.split("\n") if line.strip().startswith("1.") or line.strip()[:2].isdigit()]
        return subtasks