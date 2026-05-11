"""
Interaction Logging

Captures every agent LLM invocation (role, prompt, response, token count,
duration) to a JSONL file. This is the *training corpus* for fine-tuning
each agent's role-specific model via the self-improving-agent pipeline.

Why JSONL: append-only, line-delimited, trivially streamable, and matches
the format self-improving-agent's dataset_builder.py expects.

How to use the logs:

  1. Run the pipeline normally for a while. Every LLM call appends a row
     to ``logs/agent_interactions.jsonl``.
  2. Use ``export_for_sia(role)`` to extract a per-role training file in
     SIA's expected schema.
  3. Hand that file to SIA's MLX + LoRA pipeline (see
     ``self-improving-agent/src/self_improving_agent/training/``).
  4. When SIA produces a fine-tuned adapter, configure the model router
     to point that role at the new model:
        MODEL_FOR_developer=mlx://path/to/adapter

The interaction log is opt-in (default off). Set ``INTERACTION_LOG_PATH``
or call ``enable_logging(path)`` to start recording. Off by default to
avoid writing user prompts to disk without explicit consent.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Interaction:
    """One LLM invocation captured for training."""

    role: str
    provider: str
    model: str
    system_message: str
    user_message: str
    response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


# ===========================================
# Logger
# ===========================================

class InteractionLog:
    """Thread-safe append-only JSONL recorder."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._enabled = self.path is not None

    @property
    def enabled(self) -> bool:
        return self._enabled and self.path is not None

    def enable(self, path: str | Path) -> None:
        self.path = Path(path)
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def record(self, interaction: Interaction) -> None:
        if not self.enabled or self.path is None:
            return
        line = json.dumps(asdict(interaction), ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(line + "\n")

    def read_all(self) -> list[Interaction]:
        if not self.path or not self.path.exists():
            return []
        out: list[Interaction] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append(Interaction(**data))
        return out


# ===========================================
# Module-level singleton + helpers
# ===========================================

_log = InteractionLog(path=os.environ.get("INTERACTION_LOG_PATH"))


def get_log() -> InteractionLog:
    return _log


def enable_logging(path: str | Path) -> None:
    _log.enable(path)


def disable_logging() -> None:
    _log.disable()


def record(
    *,
    role: str,
    provider: str,
    model: str,
    system_message: str,
    user_message: str,
    response: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    duration_seconds: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Convenience: append an interaction to the module-level log."""
    _log.record(Interaction(
        role=role,
        provider=provider,
        model=model,
        system_message=system_message,
        user_message=user_message,
        response=response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_seconds=duration_seconds,
        metadata=metadata or {},
    ))


# ===========================================
# Export to self-improving-agent format
# ===========================================

def export_for_sia(
    role: str,
    out_path: str | Path,
    *,
    min_response_chars: int = 30,
) -> int:
    """Extract logged interactions for ``role`` and write SIA-format JSONL.

    SIA's dataset_builder expects per-row JSONL like:
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ], "metadata": {...}}

    We filter out rows whose response is unusably short (template stubs,
    API errors). Returns the number of rows exported.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out, "w") as f:
        for interaction in get_log().read_all():
            if interaction.role != role:
                continue
            if len(interaction.response.strip()) < min_response_chars:
                continue
            row = {
                "messages": [
                    {"role": "system", "content": interaction.system_message},
                    {"role": "user", "content": interaction.user_message},
                    {"role": "assistant", "content": interaction.response},
                ],
                "metadata": {
                    "provider": interaction.provider,
                    "model": interaction.model,
                    "timestamp": interaction.timestamp,
                    "tokens": interaction.prompt_tokens + interaction.completion_tokens,
                    **interaction.metadata,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
