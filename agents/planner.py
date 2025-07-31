# agents/planner.py

import os
import re
import openai
from openai.error import OpenAIError
from dotenv import load_dotenv
load_dotenv()

from core.memory import Memory
from core.task_schema import Task
memory = Memory("output/memory.json")

class PlannerAgent:
    def __init__(self, model="gpt-4o", temperature=0.3):
        self.model = model
        self.temperature = temperature
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

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[system_message, user_message],
                temperature=self.temperature
            )
            output = response.choices[0].message.content
        except OpenAIError as e:
            return [f"OpenAI API error: {str(e)}"]

        subtasks = [
            re.sub(r"^\s*\d+[\.\):\-]\s*", "", line).strip()
            for line in output.split("\n")
            if re.match(r"^\s*\d+[\.\):\-]", line)
        ]
        return subtasks

    def plan(self, user_prompt):
        cache_key = f"plan::{user_prompt}"
        cached = memory.get(cache_key)
        if cached:
            return [Task(**t) if isinstance(t, dict) else t for t in cached]

        result = self.plan_task(user_prompt)
        tasks = [Task(id=i, description=desc) for i, desc in enumerate(result)]
        memory.set(cache_key, tasks)
        return tasks