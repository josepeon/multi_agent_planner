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

    def evaluate_code(self, code_result, temperature: float = 0.2, max_tokens: int = 512) -> dict:
        if isinstance(code_result, dict) and code_result.get("status") == "passed":
            return code_result
        code_string = code_result.get("code", "") if isinstance(code_result, dict) else str(code_result)
        cached = self.memory.get(code_string)
        if cached:
            return cached

        try:
            prompt = (
                "You are a code reviewer. Your task is to simulate a QA evaluation of the following code. "
                "First, check if the code would run successfully if executed (assume it is in a correct environment), "
                "then provide a clear critique including any improvements, security risks, or bugs. "
                "Respond with a JSON like: "
                "{ \"success\": true/false, \"critique\": \"your comments here\" }\n\n"
                f"CODE:\n{code_string}"
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

            result = {
                "status": "complete" if parsed.get("success", False) else "failed",
                "critique": parsed.get("critique", "")
            }

            self.memory.set(code_string, result)
            return result
        except Exception as e:
            return {
                "status": "failed",
                "critique": str(e)
            }