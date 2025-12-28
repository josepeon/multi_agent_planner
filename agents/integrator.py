# agents/integrator.py
"""
Integrator Agent Module

Intelligently merges code from multiple tasks into a cohesive final program.
Unlike the simple assembler, this agent:
- Uses LLM to understand code relationships
- Resolves import conflicts
- Ensures consistent interfaces
- Validates the final output
- Supports multi-file output for larger projects
"""

import ast
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from core.llm_provider import get_llm_client, BaseLLMClient
from core.shared_context import get_shared_context, SharedContext


class IntegratorAgent:
    """Agent responsible for merging code from multiple tasks into cohesive programs."""
    
    temperature: float
    client: BaseLLMClient
    shared_context: SharedContext
    
    def __init__(self, temperature: float = 0.1) -> None:
        self.temperature = temperature
        self.client = get_llm_client(temperature=temperature)
        self.shared_context = get_shared_context()

    def integrate(self, session_log: Dict[str, Any]) -> str:
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
    def integrate_multifile(self, session_log: Dict, output_dir: str = "output/project") -> Dict[str, str]:
        """
        Integrate code into multiple files for better project structure.
        
        Args:
            session_log: The session log containing all task results
            output_dir: Directory to save the generated files
            
        Returns:
            Dictionary mapping filenames to their contents
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
            return {"main.py": "# No code generated"}
        
        # Build the code blocks section
        blocks_str = ""
        for i, block in enumerate(code_blocks, 1):
            blocks_str += f"\n### Task {i}: {block['task']}\n```python\n{block['code']}\n```\n"
        
        system_message = """You are a senior Python developer organizing code into a proper multi-file project structure.

RULES:
1. Organize code into appropriate files based on their purpose:
   - models.py: Data classes, dataclasses, Pydantic models, Enums
   - services.py: Business logic, service classes
   - utils.py: Helper functions, utilities, CLI classes
   - main.py: Entry point with main() function
2. CRITICAL: Add ALL necessary imports between files:
   - If services.py uses TaskStatus from models.py, add: "from models import Task, TaskStatus"
   - If utils.py uses TodoList from services.py, add: "from services import TodoList"
   - Import EVERY class/enum/function that is used from another file
3. Each file must be syntactically valid Python
4. The main.py should import from other modules and have a main() function

OUTPUT FORMAT (MUST follow exactly):
```json
{
  "files": {
    "models.py": "# models.py\\nfrom dataclasses import dataclass\\nfrom enum import Enum\\n...",
    "services.py": "# services.py\\nfrom models import Task, TaskStatus\\n...",
    "main.py": "# main.py\\nfrom services import ...\\ndef main():\\n    ...\\nif __name__ == '__main__':\\n    main()"
  }
}
```

Only include files that are needed. For simple projects, just use main.py.
Output ONLY the JSON, no explanation."""

        user_message = f"""Original Request: {session_log.get("prompt", "")}

Code blocks to organize:
{blocks_str}

Create the multi-file project structure as JSON:"""

        try:
            output = self.client.chat(
                user_message=user_message,
                system_message=system_message,
                temperature=self.temperature,
                max_tokens=4000
            )
            
            # Parse JSON output
            files = self._parse_multifile_output(output)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save files
            for filename, content in files.items():
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w") as f:
                    f.write(content)
                print(f"  📄 Created: {filepath}")
            
            # Validate imports between files
            self._validate_multifile_imports(files, output_dir)
            
            return files
            
        except Exception as e:
            print(f"  Multi-file integration error: {e}")
            # Fallback to single file
            single_file = self.integrate(session_log)
            return {"main.py": single_file}
    
    def _validate_multifile_imports(self, files: Dict[str, str], output_dir: str):
        """Validate and fix imports between generated files."""
        # Collect all defined classes/functions per file
        definitions = {}  # {filename: {class_names, function_names}}
        
        for filename, content in files.items():
            if not filename.endswith(".py"):
                continue
            try:
                tree = ast.parse(content)
                defs = {"classes": set(), "functions": set(), "enums": set()}
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if it's an Enum
                        for base in node.bases:
                            if isinstance(base, ast.Name) and base.id == "Enum":
                                defs["enums"].add(node.name)
                                break
                        else:
                            defs["classes"].add(node.name)
                    elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                        defs["functions"].add(node.name)
                definitions[filename] = defs
            except:
                pass
        
        # Check each file for missing imports
        for filename, content in files.items():
            if not filename.endswith(".py"):
                continue
            
            missing_imports = []
            
            # Check what names are used but not defined locally
            try:
                tree = ast.parse(content)
                local_defs = definitions.get(filename, {"classes": set(), "functions": set(), "enums": set()})
                local_names = local_defs["classes"] | local_defs["functions"] | local_defs["enums"]
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        name = node.id
                        # Skip builtins and already defined
                        if name in local_names or name in dir(__builtins__):
                            continue
                        
                        # Check if defined in another file
                        for other_file, other_defs in definitions.items():
                            if other_file == filename:
                                continue
                            all_defs = other_defs["classes"] | other_defs["functions"] | other_defs["enums"]
                            if name in all_defs:
                                module = other_file.replace(".py", "")
                                missing_imports.append((module, name))
                
                # Add missing imports
                if missing_imports:
                    # Group by module
                    by_module = {}
                    for module, name in set(missing_imports):
                        by_module.setdefault(module, set()).add(name)
                    
                    # Build import statements
                    import_lines = []
                    for module, names in by_module.items():
                        import_lines.append(f"from {module} import {', '.join(sorted(names))}")
                    
                    # Check if imports already exist
                    for imp in import_lines:
                        if imp not in content:
                            print(f"  ⚠️ Adding missing import to {filename}: {imp}")
                            # Insert after first line (comment) or at top
                            lines = content.split("\n")
                            insert_pos = 1 if lines[0].startswith("#") else 0
                            lines.insert(insert_pos, imp)
                            content = "\n".join(lines)
                            
                            # Save fixed file
                            filepath = os.path.join(output_dir, filename)
                            with open(filepath, "w") as f:
                                f.write(content)
                            files[filename] = content
            except:
                pass
    
    def _parse_multifile_output(self, output: str) -> Dict[str, str]:
        """Parse the LLM output to extract file contents."""
        output = output.strip()
        
        # Remove markdown code fences if present
        if "```json" in output:
            start = output.find("```json") + 7
            end = output.rfind("```")
            output = output[start:end].strip()
        elif "```" in output:
            start = output.find("```") + 3
            end = output.rfind("```")
            output = output[start:end].strip()
        
        try:
            data = json.loads(output)
            files = data.get("files", {})
            
            # Validate each file has valid Python syntax
            validated_files = {}
            for filename, content in files.items():
                # Unescape newlines if they were escaped
                content = content.replace("\\n", "\n")
                
                if filename.endswith(".py"):
                    try:
                        ast.parse(content)
                        validated_files[filename] = content
                    except SyntaxError as e:
                        print(f"  ⚠️ Syntax error in {filename}: {e}")
                        validated_files[filename] = f"# Syntax error in generated code\n# {e}\n\n{content}"
                else:
                    validated_files[filename] = content
            
            return validated_files if validated_files else {"main.py": "# No valid files generated"}
            
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return {"main.py": "# Failed to parse multi-file output"}