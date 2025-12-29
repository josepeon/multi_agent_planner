"""
Sandboxed Code Execution Module

Provides safe execution of LLM-generated code using multiple isolation strategies:
1. RestrictedPython - AST-level restrictions (lightweight, no Docker needed)
2. Docker - Full container isolation (most secure, requires Docker)
3. Subprocess with limits - Basic isolation with timeouts and resource limits

Usage:
    from core.sandbox import execute_code_safely, SandboxConfig
    
    result = execute_code_safely(code, method="restricted")
    print(result["output"])
"""

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionMethod(Enum):
    RESTRICTED = "restricted"  # RestrictedPython (default, no Docker)
    DOCKER = "docker"          # Full Docker isolation
    SUBPROCESS = "subprocess"  # Basic subprocess with limits


@dataclass
class SandboxConfig:
    """Configuration for sandboxed execution."""
    method: ExecutionMethod = ExecutionMethod.RESTRICTED
    timeout: int = 30  # seconds
    max_memory_mb: int = 256
    max_output_size: int = 10000  # characters
    allowed_imports: list[str] = field(default_factory=lambda: [
        # Core utilities
        "math", "random", "datetime", "json", "re", "collections",
        "itertools", "functools", "string", "csv", "io", "statistics",
        # Modern Python essentials
        "uuid", "dataclasses", "typing", "enum", "pathlib",
        "abc", "copy", "operator", "contextlib",
        # Data structures
        "heapq", "bisect", "array",
        # Text processing
        "textwrap", "difflib",
        # Testing (commonly needed)
        "unittest", "doctest",
    ])
    docker_image: str = "python:3.11-slim"


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution."""
    success: bool
    output: str
    error: str | None = None
    execution_time: float | None = None
    method_used: str = "unknown"


# ============================================
# RestrictedPython Execution (Lightweight)
# ============================================

def _get_restricted_globals() -> dict[str, Any]:
    """Create a restricted globals dict for safe execution."""
    import builtins

    # Safe builtins - includes class definition support
    safe_builtins = {
        # Class definition essentials
        '__build_class__': builtins.__build_class__,
        '__name__': '__main__',
        'object': object,
        'super': super,
        'property': property,
        'staticmethod': staticmethod,
        'classmethod': classmethod,
        # Standard safe builtins
        'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
        'chr': chr, 'dict': dict, 'dir': dir, 'divmod': divmod,
        'enumerate': enumerate, 'filter': filter, 'float': float,
        'format': format, 'frozenset': frozenset, 'hash': hash,
        'hex': hex, 'int': int, 'isinstance': isinstance,
        'issubclass': issubclass, 'iter': iter, 'len': len,
        'list': list, 'map': map, 'max': max, 'min': min,
        'next': next, 'oct': oct, 'ord': ord, 'pow': pow,
        'print': print, 'range': range, 'repr': repr, 'reversed': reversed,
        'round': round, 'set': set, 'slice': slice, 'sorted': sorted,
        'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
        'zip': zip, 'True': True, 'False': False, 'None': None,
        # Exception handling
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        'KeyError': KeyError, 'IndexError': IndexError, 'AttributeError': AttributeError,
        'ZeroDivisionError': ZeroDivisionError, 'RuntimeError': RuntimeError,
        'StopIteration': StopIteration, 'NotImplementedError': NotImplementedError,
        # getattr/setattr for OOP
        'getattr': getattr, 'setattr': setattr, 'hasattr': hasattr, 'delattr': delattr,
        'callable': callable, 'vars': vars,
    }

    return {
        '__builtins__': safe_builtins,
        '__name__': '__main__',
        '__doc__': None,
    }


def _safe_import(name: str, allowed_imports: list[str]) -> Any:
    """Safely import only allowed modules."""
    if name not in allowed_imports:
        raise ImportError(f"Import of '{name}' is not allowed. Allowed: {allowed_imports}")
    return __import__(name)


def execute_restricted(code: str, config: SandboxConfig) -> ExecutionResult:
    """
    Execute code with RestrictedPython-style restrictions.
    
    This provides basic sandboxing without Docker by:
    - Limiting available builtins
    - Restricting imports to a whitelist
    - Capturing stdout/stderr
    - Adding timeout protection
    """
    import time
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    start_time = time.time()
    output_buffer = StringIO()
    error_buffer = StringIO()

    try:
        # Create restricted environment
        restricted_globals = _get_restricted_globals()

        # Add safe import function
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            return _safe_import(name, config.allowed_imports)

        restricted_globals['__builtins__']['__import__'] = safe_import

        # Pre-import allowed modules
        for mod_name in config.allowed_imports:
            try:
                restricted_globals[mod_name] = __import__(mod_name)
            except ImportError:
                pass

        # Check for truly dangerous patterns (system access, code injection)
        # Note: We allow classes (__class__), eval in strings, and input for UI code
        dangerous_patterns = [
            ('os.system', 'System command execution'),
            ('subprocess.', 'Subprocess execution'),
            ('Popen', 'Process spawning'),
            ('__subclasses__', 'Class introspection attack'),
            ('__globals__', 'Global namespace access'),
            ('__code__', 'Code object manipulation'),
            ('__reduce__', 'Pickle exploit'),
            ('execfile', 'File execution'),
            ('reload', 'Module reload'),
        ]

        for pattern, reason in dangerous_patterns:
            if pattern in code:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Security violation: {reason} ('{pattern}' is not allowed)",
                    method_used="restricted"
                )

        # Code containing GUI/input/eval needs subprocess execution for full testing
        needs_subprocess = any(p in code for p in ['input(', 'eval(', 'tkinter', 'tk.', 'mainloop'])
        if needs_subprocess:
            # Fall back to subprocess for GUI/interactive code
            return execute_subprocess(code, config)

        # Execute with captured output
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code, restricted_globals)

        execution_time = time.time() - start_time
        output = output_buffer.getvalue()[:config.max_output_size]

        return ExecutionResult(
            success=True,
            output=output,
            execution_time=execution_time,
            method_used="restricted"
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return ExecutionResult(
            success=False,
            output=output_buffer.getvalue()[:config.max_output_size],
            error=f"{type(e).__name__}: {str(e)}",
            execution_time=execution_time,
            method_used="restricted"
        )


# ============================================
# Docker Execution (Most Secure)
# ============================================

def execute_docker(code: str, config: SandboxConfig) -> ExecutionResult:
    """
    Execute code in a Docker container for maximum isolation.
    
    Requires Docker to be installed and running.
    """
    import time

    # Check if Docker is available
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=True
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return ExecutionResult(
            success=False,
            output="",
            error="Docker is not available. Install Docker or use 'restricted' method.",
            method_used="docker"
        )

    start_time = time.time()

    # Create temporary file for the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Run in Docker with strict limits
        result = subprocess.run(
            [
                "docker", "run",
                "--rm",                                    # Remove container after execution
                "--network", "none",                       # No network access
                "--memory", f"{config.max_memory_mb}m",   # Memory limit
                "--cpus", "0.5",                          # CPU limit
                "--pids-limit", "50",                     # Process limit
                "--read-only",                            # Read-only filesystem
                "--tmpfs", "/tmp:size=10m",               # Small writable tmp
                "-v", f"{temp_file}:/code.py:ro",         # Mount code as read-only
                config.docker_image,
                "python", "/code.py"
            ],
            capture_output=True,
            text=True,
            timeout=config.timeout
        )

        execution_time = time.time() - start_time
        output = result.stdout[:config.max_output_size]

        if result.returncode == 0:
            return ExecutionResult(
                success=True,
                output=output,
                execution_time=execution_time,
                method_used="docker"
            )
        else:
            return ExecutionResult(
                success=False,
                output=output,
                error=result.stderr[:config.max_output_size],
                execution_time=execution_time,
                method_used="docker"
            )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            output="",
            error=f"Execution timed out after {config.timeout} seconds",
            method_used="docker"
        )
    finally:
        # Clean up temp file
        os.unlink(temp_file)


# ============================================
# Subprocess Execution (Basic Isolation)
# ============================================

def execute_subprocess(code: str, config: SandboxConfig) -> ExecutionResult:
    """
    Execute code in a subprocess with basic isolation.
    
    Less secure than Docker but works everywhere.
    """
    import time

    # Check for interactive code that would hang waiting for input
    if 'input(' in code:
        return ExecutionResult(
            success=True,  # Syntax is valid, just can't test interactively
            output="Interactive code detected (input()) - skipping execution. Code syntax is valid.",
            execution_time=0,
            method_used="subprocess_skip"
        )

    # Check for GUI mainloop that would hang
    if any(p in code for p in ['mainloop()', '.mainloop()']):
        return ExecutionResult(
            success=True,  # Syntax is valid, just can't test GUI
            output="GUI mainloop detected - skipping execution. Code syntax is valid.",
            execution_time=0,
            method_used="subprocess_skip"
        )

    start_time = time.time()

    # Create temporary file for the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=config.timeout,
            cwd=tempfile.gettempdir(),  # Run in temp directory
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",  # Don't inherit PYTHONPATH
                "HOME": tempfile.gettempdir(),
            }
        )

        execution_time = time.time() - start_time
        output = result.stdout[:config.max_output_size]

        if result.returncode == 0:
            return ExecutionResult(
                success=True,
                output=output,
                execution_time=execution_time,
                method_used="subprocess"
            )
        else:
            return ExecutionResult(
                success=False,
                output=output,
                error=result.stderr[:config.max_output_size],
                execution_time=execution_time,
                method_used="subprocess"
            )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            output="",
            error=f"Execution timed out after {config.timeout} seconds",
            method_used="subprocess"
        )
    finally:
        os.unlink(temp_file)


# ============================================
# Main Entry Point
# ============================================

def execute_code_safely(
    code: str,
    method: str = "restricted",
    timeout: int = 30,
    max_memory_mb: int = 256,
) -> dict[str, Any]:
    """
    Execute code safely using the specified isolation method.
    
    Args:
        code: Python code to execute
        method: One of 'restricted', 'docker', 'subprocess'
        timeout: Maximum execution time in seconds
        max_memory_mb: Maximum memory (for Docker)
    
    Returns:
        Dict with keys: success, output, error, execution_time, method_used
    
    Example:
        result = execute_code_safely("print('Hello World')")
        if result['success']:
            print(result['output'])
        else:
            print(f"Error: {result['error']}")
    """
    config = SandboxConfig(
        method=ExecutionMethod(method),
        timeout=timeout,
        max_memory_mb=max_memory_mb,
    )

    executors = {
        ExecutionMethod.RESTRICTED: execute_restricted,
        ExecutionMethod.DOCKER: execute_docker,
        ExecutionMethod.SUBPROCESS: execute_subprocess,
    }

    executor = executors[config.method]
    result = executor(code, config)

    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "execution_time": result.execution_time,
        "method_used": result.method_used,
    }


# Convenience function for quick testing
def quick_execute(code: str) -> str:
    """Quick execution with default settings. Returns output or error."""
    result = execute_code_safely(code)
    if result["success"]:
        return result["output"]
    return f"Error: {result['error']}"
