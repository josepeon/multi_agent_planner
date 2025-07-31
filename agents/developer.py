# agents/developer.py

import os
import openai
from openai import OpenAIError
from dotenv import load_dotenv
from core.memory import Memory
from core.task_schema import Task
import re
load_dotenv()

def clean_code_block(code: str) -> str:
    # Remove triple backtick Markdown formatting if present
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n", "", code)  # Remove opening triple backticks with language
        code = re.sub(r"\n```$", "", code)           # Remove closing triple backticks
    return code.strip()

class DeveloperAgent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.memory = Memory("memory/developer_memory.json")

    def write_code(self, task_description, feedback_message=None, temperature=0.3, max_tokens=500):
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

        cache_key = f"{task_description}|{feedback_message}"
        cached_code = self.memory.get(cache_key)
        if cached_code:
            return cached_code

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[system_message, user_message],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            code = response.choices[0].message.content.strip()
            code = clean_code_block(code)
            # Optionally remove markdown-like instructional content
            code = code.split("```")[0].split("To execute")[0].strip()
            self.memory.set(cache_key, code)
            return code
        except OpenAIError as e:
            return f"OpenAI API error: {str(e)}"
    def revise_code(self, task: Task, previous_code: str, feedback_message: str) -> str:
        revision_prompt = (
            f"The previous code failed with the following error or feedback:\n{feedback_message}\n\n"
            f"Please revise the following code to fix the issue:\n\n{previous_code}\n\n"
            f"Task: {task.description}"
        )

        messages = [
            {"role": "system", "content": "You are a senior Python developer. Revise the code to address the critique."},
            {"role": "user", "content": revision_prompt}
        ]

        cache_key = f"revise|{task.description}|{feedback_message}"
        cached_code = self.memory.get(cache_key)
        if cached_code:
            return cached_code

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=600,
            )
            revised_code = response.choices[0].message.content.strip()
            revised_code = clean_code_block(revised_code)
            self.memory.set(cache_key, revised_code)
            return revised_code
        except OpenAIError as e:
            return f"OpenAI API error during revision: {str(e)}"

    def develop(self, task: Task, critic=None) -> dict:
        task_code = self.write_code(task.description)
        with open("generated_code.py", "w") as f:
            f.write(task_code)

        import subprocess
        try:
            result = subprocess.run(["python", "generated_code.py"], capture_output=True, text=True)
            output = result.stdout.strip()
            passed = result.returncode == 0
        except Exception as e:
            output = str(e)
            passed = False

        if not passed and critic:
            feedback = critic.review(task.description, task_code, output)
            revised_code = self.revise_code(task, task_code, feedback)
            with open("generated_code.py", "w") as f:
                f.write(revised_code)
            try:
                result = subprocess.run(["python", "generated_code.py"], capture_output=True, text=True)
                return {
                    "code": revised_code,
                    "result": result.stdout.strip(),
                    "status": "passed" if result.returncode == 0 else "failed"
                }
            except Exception as e:
                return {
                    "code": revised_code,
                    "result": str(e),
                    "status": "failed"
                }

        return {"code": task_code, "result": output, "status": "passed" if passed else "failed"}