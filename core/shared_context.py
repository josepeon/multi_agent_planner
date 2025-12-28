"""
Shared Context Memory Module

Provides a centralized context that all agents can read from and write to.
This enables:
- Code awareness between tasks (no duplicate classes)
- Architecture sharing
- Interface consistency

Usage:
    from core.shared_context import SharedContext
    
    context = SharedContext()
    context.add_generated_code("models", "class Task: ...")
    context.get_context_summary()  # Returns summary for LLM prompts
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import os


@dataclass
class CodeBlock:
    """Represents a generated code block with metadata."""
    name: str  # e.g., "Task class", "add_task method"
    code: str
    task_id: int
    status: str  # "passed", "failed"
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)


@dataclass 
class Architecture:
    """High-level architecture specification."""
    description: str = ""
    files: List[str] = field(default_factory=list)
    classes: Dict[str, List[str]] = field(default_factory=dict)  # class_name -> [attributes]
    interfaces: Dict[str, str] = field(default_factory=dict)  # method_signature -> description
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # component -> [depends_on]


class SharedContext:
    """
    Centralized context shared across all agents.
    
    Stores:
    - Architecture specification (from Architect agent)
    - Generated code blocks (from Developer agent)
    - Interface definitions
    - Task dependencies
    """
    
    def __init__(self, filepath: str = "output/shared_context.json"):
        self.filepath = filepath
        self.architecture: Architecture = Architecture()
        self.code_blocks: Dict[int, CodeBlock] = {}  # task_id -> CodeBlock
        self.defined_classes: Dict[str, str] = {}  # class_name -> code
        self.defined_functions: Dict[str, str] = {}  # func_name -> code
        self.imports: set = set()
        self._load()
    
    def _load(self):
        """Load context from file if exists."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.imports = set(data.get("imports", []))
                    self.defined_classes = data.get("defined_classes", {})
                    self.defined_functions = data.get("defined_functions", {})
            except (json.JSONDecodeError, IOError):
                pass
    
    def _save(self):
        """Persist context to file."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        data = {
            "imports": list(self.imports),
            "defined_classes": self.defined_classes,
            "defined_functions": self.defined_functions,
            "architecture": {
                "description": self.architecture.description,
                "files": self.architecture.files,
                "classes": self.architecture.classes,
                "interfaces": self.architecture.interfaces,
            }
        }
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def set_architecture(self, architecture: Architecture):
        """Set the high-level architecture (from Architect agent)."""
        self.architecture = architecture
        self._save()
    
    def add_generated_code(self, task_id: int, name: str, code: str, status: str):
        """Add a generated code block from a task."""
        # Parse code to extract classes and functions
        classes, functions = self._extract_definitions(code)
        
        block = CodeBlock(
            name=name,
            code=code,
            task_id=task_id,
            status=status,
            classes=classes,
            functions=functions,
        )
        self.code_blocks[task_id] = block
        
        # Track defined classes/functions
        for cls in classes:
            self.defined_classes[cls] = code
        for func in functions:
            self.defined_functions[func] = code
        
        # Extract imports
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                self.imports.add(line)
        
        self._save()
    
    def _extract_definitions(self, code: str) -> tuple:
        """Extract class and function names from code using AST."""
        import ast
        classes = []
        functions = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
        except SyntaxError:
            pass
        
        return classes, functions
    
    def _extract_class_definition(self, code: str, class_name: str) -> Optional[str]:
        """Extract a specific class definition from code."""
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    return ast.get_source_segment(code, node)
        except SyntaxError:
            pass
        return None
    
    def _extract_function_signature(self, code: str, func_name: str) -> Optional[str]:
        """Extract function signature (def line) from code."""
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    # Get just the first line (signature)
                    source = ast.get_source_segment(code, node)
                    if source:
                        first_line = source.split('\n')[0]
                        # Clean up and return
                        return first_line.strip().rstrip(':')
        except SyntaxError:
            pass
        return None
    
    def get_defined_classes(self) -> List[str]:
        """Get list of all defined class names."""
        return list(self.defined_classes.keys())
    
    def get_defined_functions(self) -> List[str]:
        """Get list of all defined function names."""
        return list(self.defined_functions.keys())
    
    def get_class_code(self, class_name: str) -> Optional[str]:
        """Get the code for a specific class."""
        return self.defined_classes.get(class_name)
    
    def has_class(self, class_name: str) -> bool:
        """Check if a class has already been defined."""
        return class_name in self.defined_classes
    
    def has_function(self, func_name: str) -> bool:
        """Check if a function has already been defined."""
        return func_name in self.defined_functions
    
    def get_context_summary(self, include_code: bool = True, max_code_lines: int = 30) -> str:
        """
        Get a summary of the current context for LLM prompts.
        This helps agents know what's already been defined.
        
        Args:
            include_code: If True, include actual code snippets (not just names)
            max_code_lines: Maximum lines per code snippet to avoid token overflow
        """
        summary_parts = []
        
        # Architecture info
        if self.architecture.description:
            summary_parts.append(f"## Architecture\n{self.architecture.description}")
        
        if self.architecture.classes:
            classes_str = "\n".join([
                f"- {cls}: {', '.join(attrs)}" 
                for cls, attrs in self.architecture.classes.items()
            ])
            summary_parts.append(f"## Planned Classes\n{classes_str}")
        
        # Already defined classes - with actual code snippets
        if self.defined_classes:
            if include_code:
                classes_code = []
                for cls_name, code in self.defined_classes.items():
                    # Extract just the class definition
                    class_code = self._extract_class_definition(code, cls_name)
                    if class_code:
                        # Truncate if too long
                        lines = class_code.split('\n')
                        if len(lines) > max_code_lines:
                            class_code = '\n'.join(lines[:max_code_lines]) + f"\n    # ... ({len(lines) - max_code_lines} more lines)"
                        classes_code.append(f"### {cls_name}\n```python\n{class_code}\n```")
                
                if classes_code:
                    summary_parts.append(f"## Already Defined Classes (DO NOT REDEFINE)\n" + "\n\n".join(classes_code))
            else:
                defined_str = ", ".join(self.defined_classes.keys())
                summary_parts.append(f"## Already Defined Classes\n{defined_str}\n(Do NOT redefine these)")
        
        # Already defined functions - with signatures
        if self.defined_functions:
            funcs = [f for f in self.defined_functions.keys() if f != 'main']
            if funcs and include_code:
                func_sigs = []
                for func_name in funcs:
                    code = self.defined_functions[func_name]
                    sig = self._extract_function_signature(code, func_name)
                    if sig:
                        func_sigs.append(f"- `{sig}`")
                if func_sigs:
                    summary_parts.append(f"## Already Defined Functions\n" + "\n".join(func_sigs))
            elif funcs:
                summary_parts.append(f"## Already Defined Functions\n{', '.join(funcs)}")
        
        # Required imports
        if self.imports:
            imports_str = "\n".join(sorted(self.imports))
            summary_parts.append(f"## Available Imports\n```python\n{imports_str}\n```")
        
        if not summary_parts:
            return "No previous context. This is the first task."
        
        return "\n\n".join(summary_parts)
    
    def get_all_code(self) -> str:
        """Get all generated code combined (for final assembly)."""
        all_imports = sorted(self.imports)
        all_classes = list(self.defined_classes.values())
        all_functions = [v for k, v in self.defined_functions.items() if k != 'main']
        
        parts = []
        if all_imports:
            parts.append("\n".join(all_imports))
        parts.extend(all_classes)
        parts.extend(all_functions)
        
        return "\n\n".join(parts)
    
    def reset(self):
        """Reset the context for a new session."""
        self.architecture = Architecture()
        self.code_blocks = {}
        self.defined_classes = {}
        self.defined_functions = {}
        self.imports = set()
        if os.path.exists(self.filepath):
            os.remove(self.filepath)


# Global shared context instance
_shared_context: Optional[SharedContext] = None


def get_shared_context() -> SharedContext:
    """Get or create the global shared context instance."""
    global _shared_context
    if _shared_context is None:
        _shared_context = SharedContext()
    return _shared_context


def reset_shared_context():
    """Reset the global shared context."""
    global _shared_context
    if _shared_context:
        _shared_context.reset()
    _shared_context = None
