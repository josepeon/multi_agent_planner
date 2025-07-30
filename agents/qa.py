# agents/qa.py

import traceback
import io
import contextlib

class QAAgent:
    def __init__(self):
        pass

    def evaluate_code(self, code: str) -> dict:
        """
        Attempts to execute the generated code and returns evaluation status and output.
        """
        result = {
            "success": False,
            "error": None,
            "output": None
        }

        try:
            local_vars = {}
            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                exec(code, {}, local_vars)
            result["success"] = True
            result["output"] = stdout_buffer.getvalue()
        except Exception:
            result["error"] = traceback.format_exc()

        return result