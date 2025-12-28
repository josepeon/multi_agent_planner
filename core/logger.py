"""
Structured Logging Module

Provides consistent logging across all agents and core modules.

Features:
- Color-coded console output
- File logging with rotation
- Structured JSON format option
- Configurable log levels
- Context-aware logging (agent name, task ID)

Usage:
    from core.logger import get_logger
    
    logger = get_logger("AgentName")
    logger.info("Processing task", task_id=1)
    logger.error("Failed to execute", error="Details here")
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict
import json


# ===========================================
# Color Codes for Console Output
# ===========================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Log levels
    DEBUG = "\033[36m"     # Cyan
    INFO = "\033[32m"      # Green
    WARNING = "\033[33m"   # Yellow
    ERROR = "\033[31m"     # Red
    CRITICAL = "\033[35m"  # Magenta
    
    # Components
    AGENT = "\033[34m"     # Blue
    TASK = "\033[36m"      # Cyan
    TIME = "\033[90m"      # Gray


# ===========================================
# Custom Formatter
# ===========================================

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors for console output.
    """
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.CRITICAL,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color based on level
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        
        # Build formatted message
        parts = [
            f"{Colors.TIME}{timestamp}{Colors.RESET}",
            f"{level_color}{record.levelname:8}{Colors.RESET}",
            f"{Colors.AGENT}[{record.name}]{Colors.RESET}",
            record.getMessage(),
        ]
        
        # Add extra fields if present
        if hasattr(record, 'task_id'):
            parts.insert(3, f"{Colors.TASK}(task:{record.task_id}){Colors.RESET}")
        
        return " ".join(parts)


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured log output.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        for key in ['task_id', 'agent', 'error', 'code', 'status']:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


# ===========================================
# Context-Aware Logger
# ===========================================

class AgentLogger:
    """
    Context-aware logger wrapper for agents.
    
    Provides structured logging with automatic context fields.
    """
    
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self._logger = logger
    
    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        """Log with extra context fields."""
        extra = {k: v for k, v in kwargs.items() if v is not None}
        self._logger.log(level, msg, extra=extra)
    
    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)
    
    # Convenience methods for common patterns
    def task_start(self, task_id: int, description: str) -> None:
        self.info(f"Starting: {description[:50]}...", task_id=task_id)
    
    def task_complete(self, task_id: int) -> None:
        self.info("Task completed", task_id=task_id, status="completed")
    
    def task_failed(self, task_id: int, error: str) -> None:
        self.error(f"Task failed: {error[:100]}", task_id=task_id, status="failed")
    
    def llm_call(self, prompt_preview: str) -> None:
        self.debug(f"LLM call: {prompt_preview[:50]}...")
    
    def code_execution(self, success: bool, output: str = "") -> None:
        if success:
            self.debug(f"Code executed successfully: {output[:50]}...")
        else:
            self.warning(f"Code execution failed: {output[:50]}...")


# ===========================================
# Logger Factory
# ===========================================

_loggers: Dict[str, AgentLogger] = {}


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> None:
    """
    Configure the logging system.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_format: Use JSON format for file output
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root.handlers = []
    
    # Console handler with colors
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColoredFormatter())
    root.addHandler(console)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s"
            ))
        root.addHandler(file_handler)


def get_logger(name: str) -> AgentLogger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically agent or module name)
        
    Returns:
        AgentLogger instance with context-aware methods
    """
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = AgentLogger(name, logger)
    return _loggers[name]


# ===========================================
# Default Configuration
# ===========================================

# Set up default logging on module import
_default_level = os.environ.get("LOG_LEVEL", "INFO")
_default_file = os.environ.get("LOG_FILE", None)
_json_format = os.environ.get("LOG_JSON", "false").lower() == "true"

setup_logging(
    level=_default_level,
    log_file=_default_file,
    json_format=_json_format,
)
