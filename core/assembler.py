import os
import re
import ast
import subprocess
import sys
from typing import List, Dict, Any, Set, Tuple, Optional
from core.memory import Memory
from core.task_schema import Task


def extract_definitions(code: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Extract class names, function names, and imports from code using AST.
    Returns (classes, functions, imports).
    """
    classes = set()
    functions = set()
    imports = set()
    
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.add(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(alias.name for alias in node.names)
                imports.add(f"from {module} import {names}")
    except SyntaxError:
        pass  # Code has syntax errors, can't parse
    
    return classes, functions, imports


def is_code_complete(code: str) -> bool:
    """Check if code is syntactically complete (no truncation)."""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        # Check for common truncation indicators
        if "unexpected EOF" in str(e) or code.rstrip().endswith(("*", "+", "-", "/", "=", ",")):
            return False
        return False


def validate_code_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate code syntax using Python's compile.
    Returns (is_valid, error_message).
    """
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def extract_and_clean_code(code_blocks: List[str]) -> str:
    """
    Combine code blocks intelligently:
    - Deduplicate classes/functions by name (keep latest/most complete)
    - Consolidate imports
    - Remove incomplete/truncated code
    - Validate final output
    """
    if not code_blocks:
        return "# No code generated\npass"
    
    # Track seen definitions to avoid duplicates
    seen_classes: Dict[str, str] = {}  # class_name -> code_block
    seen_functions: Dict[str, str] = {}  # func_name -> code_block
    all_imports: Set[str] = set()
    other_code: List[str] = []
    
    for block in code_blocks:
        if not block or not block.strip():
            continue
            
        # Skip incomplete/truncated code
        if not is_code_complete(block):
            print(f"⚠️  Skipping incomplete code block (truncated)")
            continue
        
        try:
            tree = ast.parse(block)
        except SyntaxError:
            print(f"⚠️  Skipping code block with syntax errors")
            continue
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    all_imports.add(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(alias.name for alias in node.names)
                all_imports.add(f"from {module} import {names}")
        
        # Process top-level definitions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # Keep the most complete version (latest wins, assuming it's refined)
                class_code = ast.get_source_segment(block, node) or ""
                if class_code:
                    seen_classes[node.name] = class_code
            elif isinstance(node, ast.FunctionDef):
                func_code = ast.get_source_segment(block, node) or ""
                if func_code and node.name != "main":  # Handle main separately
                    seen_functions[node.name] = func_code
                elif node.name == "main":
                    # Keep the most complete main function
                    seen_functions["main"] = func_code
            elif isinstance(node, (ast.Expr, ast.Assign)):
                # Top-level expressions/assignments (e.g., root = tk.Tk())
                stmt_code = ast.get_source_segment(block, node) or ""
                if stmt_code and stmt_code not in other_code:
                    other_code.append(stmt_code)
            elif isinstance(node, ast.If):
                # Handle if __name__ == "__main__"
                if_code = ast.get_source_segment(block, node) or ""
                if "if __name__" in if_code and if_code not in other_code:
                    other_code.append(if_code)
    
    # Build final code
    parts = []
    
    # 1. Sorted imports
    if all_imports:
        # Separate standard library from third-party
        stdlib_imports = sorted([i for i in all_imports if i.startswith("import ") or "from " in i])
        parts.append("\n".join(stdlib_imports))
    
    # 2. Classes
    for class_name, class_code in seen_classes.items():
        parts.append(class_code)
    
    # 3. Functions (except main)
    for func_name, func_code in seen_functions.items():
        if func_name != "main":
            parts.append(func_code)
    
    # 4. Main function last
    if "main" in seen_functions:
        parts.append(seen_functions["main"])
    
    # 5. Other top-level code (like tkinter mainloop)
    # Filter out duplicates and GUI mainloop if we have a main function
    has_main = "main" in seen_functions
    for code in other_code:
        if has_main and "mainloop()" in code:
            continue  # Skip standalone mainloop if we have a main function
        if "if __name__" in code:
            continue  # Skip if __name__ blocks, we'll add one
        parts.append(code)
    
    # 6. Add if __name__ == "__main__" block if we have a main function
    if "main" in seen_functions:
        parts.append('if __name__ == "__main__":\n    main()')
    
    final_code = "\n\n\n".join(parts)
    
    # Final validation
    is_valid, error = validate_code_syntax(final_code)
    if not is_valid:
        print(f"⚠️  Warning: Final code has syntax errors: {error}")
    
    return final_code


def save_final_script(code: str, path: str = "output/final_program.py") -> None:
    """Save the final script and optionally run a syntax check."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(code)
    
    # Verify with Python syntax check
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", path],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"\n✅ Final program saved to: {path}")
        print("✅ Syntax validation passed")
    else:
        print(f"\n⚠️  Final program saved to: {path}")
        print(f"⚠️  Syntax validation failed: {result.stderr}")


def assemble_code_from_log(log_data: Dict[str, Any]) -> str:
    """
    Assemble final code from session log.
    Only includes code from tasks that passed QA or have valid code.
    """
    code_blocks = []
    
    for task in log_data.get("tasks", []):
        code = task.get("code")
        status = task.get("status", "")
        qa_result = task.get("qa_result", {})
        
        # Skip if no code
        if not code or not isinstance(code, str):
            continue
        
        # Skip placeholder code
        if code.strip().startswith("# Create a file"):
            continue
            
        # Prefer code that passed QA, but include others if they're syntactically valid
        qa_passed = qa_result.get("status") == "passed" if qa_result else False
        
        if qa_passed or is_code_complete(code):
            code_blocks.append(code)
    
    # Deduplicate identical blocks
    seen = set()
    unique_blocks = []
    for block in code_blocks:
        block_hash = hash(block.strip())
        if block_hash not in seen:
            unique_blocks.append(block)
            seen.add(block_hash)
    
    final_code = extract_and_clean_code(unique_blocks)
    
    # Save to memory
    memory = Memory(filepath="output/memory.json")
    memory.set("final_code", final_code)
    
    save_final_script(final_code)
    return final_code