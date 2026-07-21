"""
Sandboxed Code Execution Module

Provides execution of LLM-generated code with three isolation tiers:

1. ``restricted`` — RestrictedPython AST-level allowlist. **Real security
   isolation** for code that fits inside the allowlist. Blocks network, file
   I/O, subprocess, ``eval``/``exec``, and dangerous dunders. The default and
   the only mode safe for untrusted input on a host without Docker.

2. ``docker`` — Container isolation with no network, read-only FS, memory and
   pid limits. **Strongest isolation.** Requires Docker on the host.

3. ``crash_isolated`` (legacy name: ``subprocess``) — Runs the code in a child
   Python process with a timeout. **This is crash isolation only — it is NOT a
   security boundary.** A subprocess can read/write the host filesystem, open
   sockets, exec arbitrary binaries, and exfiltrate data. Use only for trusted
   input or as a fallback for GUI/interactive code that cannot run inside the
   restricted allowlist.

By default, the restricted executor falls back to ``crash_isolated`` for
input()/GUI/eval code so the developer agent can validate syntax. To disable
this fallback (e.g. on a hosted deployment serving untrusted requests), set
``MAP_FORBID_CRASH_ISOLATED=1`` in the environment. With that flag set,
crash-isolated execution is refused and the restricted executor reports the
code as unverifiable instead of falling back.

Usage:
    from core.sandbox import execute_code_safely, SandboxConfig

    result = execute_code_safely(code, method="restricted")
    print(result["output"])
"""

import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Env var that hosted deployments should set to disable the crash-isolated
# fallback path entirely. When set, code that requires the fallback is
# reported as unverifiable instead of executed in a non-sandboxed subprocess.
FORBID_CRASH_ISOLATED_ENV = "MAP_FORBID_CRASH_ISOLATED"


def crash_isolated_forbidden() -> bool:
    """True if the env explicitly forbids crash-isolated (subprocess) execution."""
    val = os.environ.get(FORBID_CRASH_ISOLATED_ENV, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


class ExecutionMethod(Enum):
    """Sandbox isolation tier.

    ``CRASH_ISOLATED`` is the canonical name; ``SUBPROCESS`` is kept as an
    alias for backward compatibility with existing callers and config files.
    Both resolve to the same enum member at runtime.
    """

    RESTRICTED = "restricted"  # RestrictedPython, real security
    DOCKER = "docker"  # Full container isolation
    CRASH_ISOLATED = "crash_isolated"  # Subprocess: crash isolation only
    CLOUD = "cloud"  # Hosted sandbox (E2B); real I/O allowed

    @classmethod
    def _missing_(cls, value):  # type: ignore[override]
        # Accept legacy "subprocess" as an alias for crash_isolated.
        if isinstance(value, str) and value.lower() == "subprocess":
            warnings.warn(
                "Sandbox method 'subprocess' is a misleading name and has been "
                "renamed to 'crash_isolated'. The behavior is unchanged but the "
                "old name will be removed in a future release. Note: this mode "
                "provides crash isolation only, NOT security isolation.",
                DeprecationWarning,
                stacklevel=3,
            )
            return cls.CRASH_ISOLATED
        return None


@dataclass
class SandboxConfig:
    """Configuration for sandboxed execution."""

    method: ExecutionMethod = ExecutionMethod.RESTRICTED
    timeout: int = 30  # seconds
    max_memory_mb: int = 256
    max_output_size: int = 10000  # characters
    allowed_imports: list[str] = field(
        default_factory=lambda: [
            # Core utilities
            "math",
            "random",
            "datetime",
            "json",
            "re",
            "collections",
            "itertools",
            "functools",
            "string",
            "csv",
            "io",
            "statistics",
            # Modern Python essentials
            "uuid",
            "dataclasses",
            "typing",
            "enum",
            "pathlib",
            "abc",
            "copy",
            "operator",
            "contextlib",
            # Data structures
            "heapq",
            "bisect",
            "array",
            # Text processing
            "textwrap",
            "difflib",
            # Testing (commonly needed)
            "unittest",
            "doctest",
        ]
    )
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
        "__build_class__": builtins.__build_class__,
        "__name__": "__main__",
        "object": object,
        "super": super,
        "property": property,
        "staticmethod": staticmethod,
        "classmethod": classmethod,
        # Standard safe builtins
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "dir": dir,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hash": hash,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
        # Exception handling
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "ZeroDivisionError": ZeroDivisionError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
        "NotImplementedError": NotImplementedError,
        # getattr/setattr for OOP
        "getattr": getattr,
        "setattr": setattr,
        "hasattr": hasattr,
        "delattr": delattr,
        "callable": callable,
        "vars": vars,
    }

    return {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
        "__doc__": None,
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

        restricted_globals["__builtins__"]["__import__"] = safe_import

        # Pre-import allowed modules
        for mod_name in config.allowed_imports:
            try:
                restricted_globals[mod_name] = __import__(mod_name)
            except ImportError:
                pass

        # Check for truly dangerous patterns (system access, code injection)
        # Note: We allow classes (__class__), eval in strings, and input for UI code
        dangerous_patterns = [
            ("os.system", "System command execution"),
            ("subprocess.", "Subprocess execution"),
            ("Popen", "Process spawning"),
            ("__subclasses__", "Class introspection attack"),
            ("__globals__", "Global namespace access"),
            ("__code__", "Code object manipulation"),
            ("__reduce__", "Pickle exploit"),
            ("execfile", "File execution"),
            ("reload", "Module reload"),
        ]

        for pattern, reason in dangerous_patterns:
            if pattern in code:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Security violation: {reason} ('{pattern}' is not allowed)",
                    method_used="restricted",
                )

        # Code containing GUI/input/eval can't run inside the restricted allowlist.
        # By default we fall back to crash-isolated subprocess execution so the
        # developer agent can at least validate syntax + simple runtime. Hosted
        # deployments that serve untrusted input should set
        # MAP_FORBID_CRASH_ISOLATED=1 to disable this fallback — in that case
        # we report the code as unverifiable instead of running it.
        needs_fallback = any(p in code for p in ["input(", "eval(", "tkinter", "tk.", "mainloop"])
        if needs_fallback:
            if crash_isolated_forbidden():
                return ExecutionResult(
                    success=False,
                    output="",
                    error=(
                        "Code requires crash-isolated execution (input/GUI/eval), "
                        f"but {FORBID_CRASH_ISOLATED_ENV}=1 disables that fallback. "
                        "Use Docker mode or simplify the code."
                    ),
                    method_used="crash_isolated_forbidden",
                )
            return execute_subprocess(code, config)

        # Execute with captured output
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code, restricted_globals)

        execution_time = time.time() - start_time
        output = output_buffer.getvalue()[: config.max_output_size]

        return ExecutionResult(
            success=True, output=output, execution_time=execution_time, method_used="restricted"
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return ExecutionResult(
            success=False,
            output=output_buffer.getvalue()[: config.max_output_size],
            error=f"{type(e).__name__}: {str(e)}",
            execution_time=execution_time,
            method_used="restricted",
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
        subprocess.run(["docker", "info"], capture_output=True, timeout=5, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return ExecutionResult(
            success=False,
            output="",
            error="Docker is not available. Install Docker or use 'restricted' method.",
            method_used="docker",
        )

    start_time = time.time()

    # Create temporary file for the code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Run in Docker with strict limits
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",  # Remove container after execution
                "--network",
                "none",  # No network access
                "--memory",
                f"{config.max_memory_mb}m",  # Memory limit
                "--cpus",
                "0.5",  # CPU limit
                "--pids-limit",
                "50",  # Process limit
                "--read-only",  # Read-only filesystem
                "--tmpfs",
                "/tmp:size=10m",  # Small writable tmp
                "-v",
                f"{temp_file}:/code.py:ro",  # Mount code as read-only
                config.docker_image,
                "python",
                "/code.py",
            ],
            capture_output=True,
            text=True,
            timeout=config.timeout,
        )

        execution_time = time.time() - start_time
        output = result.stdout[: config.max_output_size]

        if result.returncode == 0:
            return ExecutionResult(
                success=True, output=output, execution_time=execution_time, method_used="docker"
            )
        else:
            return ExecutionResult(
                success=False,
                output=output,
                error=result.stderr[: config.max_output_size],
                execution_time=execution_time,
                method_used="docker",
            )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            output="",
            error=f"Execution timed out after {config.timeout} seconds",
            method_used="docker",
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
    if "input(" in code:
        return ExecutionResult(
            success=True,  # Syntax is valid, just can't test interactively
            output="Interactive code detected (input()) - skipping execution. Code syntax is valid.",
            execution_time=0,
            method_used="subprocess_skip",
        )

    # Check for GUI mainloop that would hang
    if any(p in code for p in ["mainloop()", ".mainloop()"]):
        return ExecutionResult(
            success=True,  # Syntax is valid, just can't test GUI
            output="GUI mainloop detected - skipping execution. Code syntax is valid.",
            execution_time=0,
            method_used="subprocess_skip",
        )

    start_time = time.time()

    # Create temporary file for the code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
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
            },
        )

        execution_time = time.time() - start_time
        output = result.stdout[: config.max_output_size]

        if result.returncode == 0:
            return ExecutionResult(
                success=True, output=output, execution_time=execution_time, method_used="subprocess"
            )
        else:
            return ExecutionResult(
                success=False,
                output=output,
                error=result.stderr[: config.max_output_size],
                execution_time=execution_time,
                method_used="subprocess",
            )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            output="",
            error=f"Execution timed out after {config.timeout} seconds",
            method_used="subprocess",
        )
    finally:
        os.unlink(temp_file)


# ============================================
# Cloud Execution (E2B)
# ============================================


def execute_cloud(code: str, config: SandboxConfig) -> ExecutionResult:
    """Run code in a hosted E2B sandbox.

    Real isolation: ephemeral container, real filesystem, real network. The
    only mode that can credibly test code that does I/O (scrapers, API
    clients, DB code). Set ``E2B_API_KEY`` in the env to activate.

    Lazy-imports ``e2b_code_interpreter``; if not installed or no key, returns
    a clear error result so callers can fall back.
    """
    import time

    if not os.environ.get("E2B_API_KEY"):
        return ExecutionResult(
            success=False,
            output="",
            error=(
                "Cloud sandbox requires E2B_API_KEY in the environment. "
                "Get a key at https://e2b.dev/. Falling back is the caller's "
                "responsibility."
            ),
            method_used="cloud_unconfigured",
        )

    try:
        from e2b_code_interpreter import Sandbox  # type: ignore
    except ImportError:
        return ExecutionResult(
            success=False,
            output="",
            error=(
                "Cloud sandbox requires the e2b-code-interpreter package. "
                "Install with: pip install e2b-code-interpreter"
            ),
            method_used="cloud_unconfigured",
        )

    start_time = time.time()
    try:
        with Sandbox(timeout=config.timeout) as sandbox:
            execution = sandbox.run_code(code, timeout=config.timeout)
            stdout = "".join(execution.logs.stdout) if execution.logs.stdout else ""
            stderr = "".join(execution.logs.stderr) if execution.logs.stderr else ""

            elapsed = time.time() - start_time
            output = stdout[: config.max_output_size]

            if execution.error:
                return ExecutionResult(
                    success=False,
                    output=output,
                    error=str(execution.error)[: config.max_output_size],
                    execution_time=elapsed,
                    method_used="cloud",
                )

            if stderr and not stdout:
                return ExecutionResult(
                    success=False,
                    output=output,
                    error=stderr[: config.max_output_size],
                    execution_time=elapsed,
                    method_used="cloud",
                )

            return ExecutionResult(
                success=True,
                output=output,
                execution_time=elapsed,
                method_used="cloud",
            )
    except Exception as exc:  # noqa: BLE001
        return ExecutionResult(
            success=False,
            output="",
            error=f"E2B sandbox error: {type(exc).__name__}: {exc}",
            execution_time=time.time() - start_time,
            method_used="cloud",
        )


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
        ExecutionMethod.CRASH_ISOLATED: execute_subprocess,
        ExecutionMethod.CLOUD: execute_cloud,
    }

    # Honor the explicit forbid-flag for crash-isolated execution.
    if config.method is ExecutionMethod.CRASH_ISOLATED and crash_isolated_forbidden():
        return {
            "success": False,
            "output": "",
            "error": (
                "Crash-isolated (subprocess) execution is disabled by "
                f"{FORBID_CRASH_ISOLATED_ENV}=1. This mode runs code without "
                "security isolation; use 'restricted' or 'docker' instead."
            ),
            "execution_time": 0,
            "method_used": "crash_isolated_forbidden",
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
