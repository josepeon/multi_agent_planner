# agents/qa.py

"""
QA Agent

Validates code by checking actual execution results from the sandbox.
If the sandbox has already run the code, trust those results.
Only use LLM for static code review when no execution data is available.
"""

import json
from core.llm_provider import get_llm_client
from core.memory import Memory
from core.task_schema import Task


class QAAgent:
    def __init__(self, temperature=0.2):
        self.temperature = temperature
        self.client = get_llm_client(temperature=temperature, max_tokens=512)
        self.memory = Memory(filepath="output/qa_memory.json")

    def evaluate_code(self, code_result, temperature: float = 0.2, max_tokens: int = 512) -> dict:
        """
        Evaluate code based on execution results.
        
        Priority:
        1. Trust actual sandbox execution results (if available)
        2. Fall back to LLM static analysis (if no execution data)
        """
        
        # If we have execution data from sandbox, trust it
        if isinstance(code_result, dict):
            status = code_result.get("status")
            result_output = code_result.get("result", "")
            
            # Trust actual execution results from sandbox
            if status in ["passed", "failed"]:
                return {
                    "status": status,
                    "result": result_output,
                    "critique": self._get_critique(code_result) if status == "failed" else "",
                }
        
        # Fallback: LLM static analysis (when no execution data)
        code_string = code_result.get("code", "") if isinstance(code_result, dict) else str(code_result)
        return self._llm_static_analysis(code_string, temperature, max_tokens)
    
    def _get_critique(self, code_result: dict) -> str:
        """Extract or generate critique for failed code."""
        result_output = code_result.get("result", "")
        code = code_result.get("code", "")
        
        # If there's an error message, that's the critique
        if "Error:" in result_output or "error:" in result_output.lower():
            return result_output
        
        return f"Code execution failed. Output: {result_output[:500]}"
    
    def _llm_static_analysis(self, code_string: str, temperature: float, max_tokens: int) -> dict:
        """Use LLM for static code analysis when no execution data is available."""
        cached = self.memory.get(code_string)
        if cached:
            return cached

        try:
            system_message = "You are a senior Python code reviewer. Always respond with valid JSON only."
            prompt = (
                "You are a code reviewer. Your task is to simulate a QA evaluation of the following code. "
                "First, check if the code would run successfully if executed (assume it is in a correct environment), "
                "then provide a clear critique including any improvements, security risks, or bugs. "
                "Respond with a JSON like: "
                '{ "success": true/false, "critique": "your comments here" }\n\n'
                f"CODE:\n{code_string}"
            )

            response = self.client.chat(
                user_message=prompt,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract JSON from response (handle markdown code blocks)
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            parsed = json.loads(response_text)

            result = {
                "status": "passed" if parsed.get("success", False) else "failed",
                "critique": parsed.get("critique", "")
            }

            self.memory.set(code_string, result)
            return result
        except Exception as e:
            return {
                "status": "failed",
                "critique": str(e)
            }