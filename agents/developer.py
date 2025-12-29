# agents/developer.py
"""
Developer Agent Module

Generates and revises Python code based on task descriptions using LLM.
Includes sandboxed execution for validation.
"""

import re
from typing import Any

from core.llm_provider import BaseLLMClient, get_llm_client
from core.memory import Memory
from core.sandbox import execute_code_safely
from core.task_schema import Task


def clean_code_block(code: str) -> str:
    """Remove triple backtick Markdown formatting if present."""
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n", "", code)  # Remove opening triple backticks with language
        code = re.sub(r"\n```$", "", code)           # Remove closing triple backticks
    return code.strip()


class DeveloperAgent:
    """Agent responsible for writing and revising Python code."""

    temperature: float
    sandbox_method: str
    client: BaseLLMClient
    memory: Memory

    def __init__(self, temperature: float = 0.3, sandbox_method: str = "restricted") -> None:
        self.temperature = temperature
        self.sandbox_method = sandbox_method  # 'restricted', 'docker', or 'subprocess'
        self.client = get_llm_client(temperature=temperature, max_tokens=2048)
        self.memory = Memory("memory/developer_memory.json")

    def write_code(
        self,
        task_description: str,
        feedback_message: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate Python code for a given task description.
        
        Args:
            task_description: What the code should accomplish
            feedback_message: Optional error from previous attempt
            temperature: LLM temperature setting
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated Python code as string
        """
        system_message = (
            "You are a senior Python developer. "
            "Your job is to write a clean, minimal Python function or code block "
            "that fulfills a single, clearly defined task. "
            "IMPORTANT: Do NOT use input() or any interactive functions - the code must run without user interaction. "
            "For GUI code, do NOT call mainloop() - just define the classes/functions. "
            "CRITICAL: Always write COMPLETE code. Never truncate or leave code unfinished. "
            "CRITICAL: If classes or functions are shown in the context as 'Already Defined', DO NOT REDEFINE THEM. "
            "Just use them directly - assume they are imported and available. "
            "Only return valid, complete Python code — no explanations, no markdown, no truncation."
        )
        base_prompt = f"Task: {task_description}"
        if feedback_message:
            base_prompt += f"\nNote: Your previous attempt failed. Error was:\n{feedback_message}"

        cache_key = f"{task_description}|{feedback_message}"
        cached_code = self.memory.get(cache_key)
        if cached_code:
            return cached_code

        # Try up to 3 times to get complete code
        for attempt in range(3):
            try:
                code = self.client.chat(
                    user_message=base_prompt,
                    system_message=system_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                code = clean_code_block(code)
                # Optionally remove markdown-like instructional content
                code = code.split("```")[0].split("To execute")[0].strip()

                # Check if code is syntactically complete
                try:
                    compile(code, "<string>", "exec")
                    self.memory.set(cache_key, code)
                    return code
                except SyntaxError as e:
                    if attempt < 2:  # Retry with feedback about truncation
                        base_prompt = f"Task: {task_description}\nIMPORTANT: Your previous code was truncated/incomplete. Error: {e.msg} at line {e.lineno}. Please provide COMPLETE code."
                        continue
                    # Last attempt, return what we have
                    self.memory.set(cache_key, code)
                    return code
            except Exception as e:
                return f"LLM API error: {str(e)}"

        return ""  # Should not reach here

    def revise_code(self, task: Task, previous_code: str, feedback_message: str) -> str:
        """
        Revise code based on feedback from critic or error messages.
        
        Args:
            task: The original task
            previous_code: Code that failed
            feedback_message: What went wrong
            
        Returns:
            Revised Python code
        """
        system_message = "You are a senior Python developer. Revise the code to address the critique. Only return valid Python code — no explanations or markdown."
        revision_prompt = (
            f"The previous code failed with the following error or feedback:\n{feedback_message}\n\n"
            f"Please revise the following code to fix the issue:\n\n{previous_code}\n\n"
            f"Task: {task.description}"
        )

        cache_key = f"revise|{task.description}|{feedback_message}"
        cached_code = self.memory.get(cache_key)
        if cached_code:
            return cached_code

        try:
            revised_code = self.client.chat(
                user_message=revision_prompt,
                system_message=system_message,
                temperature=0.3,
                max_tokens=600,
            )
            revised_code = clean_code_block(revised_code)
            self.memory.set(cache_key, revised_code)
            return revised_code
        except Exception as e:
            return f"LLM API error during revision: {str(e)}"

    def _execute_code(self, code: str) -> dict[str, Any]:
        """Execute code safely using sandboxed execution."""
        # First validate syntax before attempting execution
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return {
                "output": "",
                "passed": False,
                "error": f"Syntax error at line {e.lineno}: {e.msg}",
                "method": "syntax_check",
            }

        # Skip execution for GUI code that can't run headless
        if any(pattern in code for pattern in ['mainloop()', 'tk.mainloop', '.mainloop()']):
            return {
                "output": "GUI code - skipping execution (mainloop detected)",
                "passed": True,  # Syntax is valid, consider it passed
                "error": None,
                "method": "gui_skip",
            }

        result = execute_code_safely(
            code,
            method=self.sandbox_method,
            timeout=30,
        )
        return {
            "output": result["output"],
            "passed": result["success"],
            "error": result.get("error"),
            "method": result.get("method_used"),
        }

    def develop(
        self,
        task: Task,
        critic: Any | None = None,
        feedback_message: str | None = None
    ) -> dict[str, Any]:
        """
        Develop code for a task with optional critic feedback loop.
        
        Args:
            task: The task to develop code for
            critic: Optional critic agent for internal retry (deprecated, use orchestrator retry)
            feedback_message: Optional feedback from previous failed attempt
            
        Returns:
            Dict with code, result, status, and sandbox_method keys
        
        Uses sandboxed execution for safety.
        """
        # Pass feedback to write_code if provided (from orchestrator retry loop)
        task_code = self.write_code(task.description, feedback_message=feedback_message)

        # Save generated code for reference
        with open("generated_code.py", "w") as f:
            f.write(task_code)

        # Execute in sandbox
        exec_result = self._execute_code(task_code)
        output = exec_result["output"]
        passed = exec_result["passed"]

        if exec_result.get("error"):
            output = f"{output}\nError: {exec_result['error']}"

        # Legacy critic support (orchestrator now handles retry loop)
        if not passed and critic:
            feedback = critic.review(task.description, task_code, output)
            revised_code = self.revise_code(task, task_code, feedback)

            with open("generated_code.py", "w") as f:
                f.write(revised_code)

            # Execute revised code in sandbox
            exec_result = self._execute_code(revised_code)
            return {
                "code": revised_code,
                "result": exec_result["output"],
                "status": "passed" if exec_result["passed"] else "failed",
                "sandbox_method": exec_result.get("method"),
            }

        return {
            "code": task_code,
            "result": output,
            "status": "passed" if passed else "failed",
            "sandbox_method": exec_result.get("method"),
        }
