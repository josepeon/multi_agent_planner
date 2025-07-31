# agents/qa.py

import os
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class QAAgent:
    def __init__(self, model="gpt-4"):
        self.model = model

    def evaluate_code(self, code: str) -> dict:
        """
        Sends the code to the OpenAI API for evaluation and critique.
        """
        result = {
            "success": False,
            "error": None,
            "output": None,
            "critique": None
        }

        try:
            prompt = (
                "You are a code reviewer. Your task is to simulate a QA evaluation of the following code. "
                "First, check if the code would run successfully if executed (assume it is in a correct environment), "
                "then provide a clear critique including any improvements, security risks, or bugs. "
                "Respond with a JSON like: "
                "{ \"success\": true/false, \"critique\": \"your comments here\" }\n\n"
                f"CODE:\n{code}"
            )

            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior Python code reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            content = response.choices[0].message.content
            import json
            parsed = json.loads(content)

            result["success"] = parsed.get("success", False)
            result["critique"] = parsed.get("critique", "")
        except Exception as e:
            result["error"] = str(e)

        return result