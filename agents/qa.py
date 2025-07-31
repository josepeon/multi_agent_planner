# agents/qa.py

import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
from core.memory import Memory
from core.task_schema import Task

load_dotenv()

class QAAgent:
    def __init__(self, model="gpt-4"):
        self.model = model
        self.memory = Memory(filepath="output/qa_memory.json")

    def evaluate_code(self, task: Task, temperature: float = 0.2, max_tokens: int = 512) -> Task:
        cached = self.memory.get(task.description)
        if cached:
            return cached

        try:
            prompt = (
                "You are a code reviewer. Your task is to simulate a QA evaluation of the following code. "
                "First, check if the code would run successfully if executed (assume it is in a correct environment), "
                "then provide a clear critique including any improvements, security risks, or bugs. "
                "Respond with a JSON like: "
                "{ \"success\": true/false, \"critique\": \"your comments here\" }\n\n"
                f"CODE:\n{task.description}"
            )

            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior Python code reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            import json
            parsed = json.loads(response.choices[0].message.content)

            if parsed.get("success", False):
                task.status = "complete"
            else:
                task.status = "failed"
            task.result = parsed.get("critique", "")

            self.memory.set(task.description, task)
        except Exception as e:
            task.status = "failed"
            task.result = str(e)

        return task