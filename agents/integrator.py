# agents/integrator.py

"""
Integrator Agent

Intelligently merges code from multiple tasks into a cohesive final program.
Unlike the simple assembler, this agent:
- Uses LLM to understand code relationships
- Resolves import conflicts
- Ensures consistent interfaces
- Validates the final output
"""

from core.llm_provider import get_llm_client
from core.shared_context import get_shared_context
from typing import List, Dict
import ast


class IntegratorAgent:
    def __init__(self, temperature=0.1):
        self.temperature = temperature
        self.client = get_llm_client(temperature=temperature)
        self.shared_context = get_shared_context()

    def integrate(self, session_log: Dict) -> str:
        """
        Integrate code from all tasks into a final cohesive program.
        
        Args:
            session_log: The session log containing all task results
            
        Returns:
            The final integrated Python code
        """
        
        # Collect all code blocks
        code_blocks = []
        for task_entry in session_log.get("tasks", []):
            code = task_entry.get("code", "")
            if code and isinstance(code, str) and code.strip():
                code_blocks.append({
                    "task": task_entry.get("task", "Unknown task"),
                    "code": code,
                    "status": task_entry.get("status", "unknown")
                })
        
        if not code_blocks:
            return "# No code generated"
        
        # If only one block, just clean it up
        if len(code_blocks) == 1:
            return self._clean_code(code_blocks[0]["code"])
        
        # Use LLM to intelligently merge
        return self._llm_merge(code_blocks, session_log.get("prompt", ""))
    
    def _llm_merge(self, code_blocks: List[Dict], original_prompt: str) -> str:
        """Use LLM to merge multiple code blocks intelligently."""
        
        # Build the code blocks section
        blocks_str = ""
        for i, block in enumerate(code_blocks, 1):
            blocks_str += f"\n### Task {i}: {block['task']}\n```python\n{block['code']}\n```\n"
        
        system_message = """You are a senior Python developer integrating code from multiple tasks into one cohesive program.

RULES:
1. Combine all code into ONE valid Python file
2. Remove duplicate imports - keep only one of each
3. Remove duplicate class/function definitions - keep the most complete version
4. Ensure consistent interfaces between components
5. Add a main() function at the end if there isn't one
6. Add if __name__ == "__main__": main() at the very end
7. Order code correctly: imports → classes → functions → main()
8. Output ONLY the Python code, no markdown, no explanation

The code should be complete and runnable."""

        user_message = f"""Original Request: {original_prompt}

Code blocks to integrate:
{blocks_str}

Create the final integrated Python program:"""

        try:
            output = self.client.chat(
                user_message=user_message,
                system_message=system_message,
                temperature=self.temperature,
                max_tokens=3000
            )
            
            # Clean up the output
            code = self._extract_code(output)
            
            # Validate syntax
            if self._validate_syntax(code):
                return code
            else:
                # Fallback to AST-based merge
                print("  LLM merge had syntax errors, using AST fallback...")
                return self._ast_merge(code_blocks)
                
        except Exception as e:
            print(f"  Integrator LLM error: {e}")
            return self._ast_merge(code_blocks)
    
    def _extract_code(self, output: str) -> str:
        """Extract Python code from LLM output."""
        output = output.strip()
        
        # Remove markdown code fences if present
        if output.startswith("```python"):
            output = output[9:]
        elif output.startswith("```"):
            output = output[3:]
        
        if output.endswith("```"):
            output = output[:-3]
        
        return output.strip()
    
    def _validate_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _clean_code(self, code: str) -> str:
        """Clean up a single code block."""
        code = self._extract_code(code)
        
        # Ensure main block
        if "if __name__" not in code:
            if "def main(" in code:
                code += '\n\nif __name__ == "__main__":\n    main()\n'
        
        return code
    
    def _ast_merge(self, code_blocks: List[Dict]) -> str:
        """Fallback: merge code blocks using AST analysis."""
        
        imports = set()
        classes = {}
        functions = {}
        other_code = []
        
        for block in code_blocks:
            code = block["code"]
            if not code or not isinstance(code, str):
                continue
                
            try:
                tree = ast.parse(code)
                
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        # Get import as string
                        import_str = ast.get_source_segment(code, node)
                        if import_str:
                            imports.add(import_str)
                    
                    elif isinstance(node, ast.ClassDef):
                        # Keep most complete version (by line count)
                        class_code = ast.get_source_segment(code, node)
                        if class_code:
                            if node.name not in classes or len(class_code) > len(classes[node.name]):
                                classes[node.name] = class_code
                    
                    elif isinstance(node, ast.FunctionDef):
                        func_code = ast.get_source_segment(code, node)
                        if func_code:
                            if node.name not in functions or len(func_code) > len(functions[node.name]):
                                functions[node.name] = func_code
                    
                    else:
                        # Other top-level code
                        segment = ast.get_source_segment(code, node)
                        if segment and segment.strip():
                            other_code.append(segment)
                            
            except SyntaxError:
                # If we can't parse, just include the raw code
                other_code.append(code)
        
        # Assemble final code
        parts = []
        
        # Imports first
        if imports:
            parts.append("\n".join(sorted(imports)))
        
        # Classes
        if classes:
            parts.extend(classes.values())
        
        # Functions (main last)
        main_func = functions.pop("main", None)
        if functions:
            parts.extend(functions.values())
        
        if main_func:
            parts.append(main_func)
        
        # Other code
        for code in other_code:
            if code.strip() and 'if __name__' not in code:
                parts.append(code)
        
        final_code = "\n\n".join(parts)
        
        # Add main block if missing
        if "if __name__" not in final_code and "def main(" in final_code:
            final_code += '\n\nif __name__ == "__main__":\n    main()\n'
        
        return final_code
