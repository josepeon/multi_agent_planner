"""
Target Language Abstraction

The pipeline was hardcoded to Python — every agent prompt said "Python",
the Developer's system message instructed Python idioms, the integrator
assumed `.py` files, the sandbox only knew RestrictedPython.

This module introduces a ``Language`` enum and a ``LanguageProfile`` that
describes the language-specific bits the agents need: prompt fragments,
file extension, default entry point, idiomatic comment style, and the
test framework name. Agents consult the active profile rather than
hardcoding Python everywhere.

Scope of this change:

- Python is fully supported (current behavior; same defaults).
- TypeScript is supported at the **scaffold** level — prompts and file
  extensions are correct, generated artifacts are valid `.ts` files —
  but the existing sandbox can't execute them (cloud/E2B mode handles
  this; restricted/docker do not). This is the "foundation" referenced
  in ROADMAP_V2.md T3.3; per-language sandbox runners come later.

Activate via ``LANGUAGE=typescript`` env, or by passing
``language=Language.TYPESCRIPT`` into ``get_profile``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class Language(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"

    @classmethod
    def _missing_(cls, value):  # type: ignore[override]
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"python", "py", "python3"}:
                return cls.PYTHON
            if v in {"typescript", "ts", "node", "javascript", "js"}:
                return cls.TYPESCRIPT
        return None


@dataclass(frozen=True)
class LanguageProfile:
    """Everything an agent needs to know to target a language."""

    name: Language
    display: str
    file_extension: str  # ".py", ".ts"
    entrypoint: str  # canonical entry file relative to project root
    test_extension: str  # ".py" for pytest, ".test.ts" for vitest etc.
    test_framework: str  # "pytest", "vitest", ...
    comment_style: str  # "#", "//"
    package_manifest: str  # "requirements.txt", "package.json"
    syntax_check_marker: str  # whatever the developer agent expects to see

    # Prompt fragments — embedded by the Developer / Integrator / TestGen agents
    developer_system_addendum: str
    test_system_addendum: str

    def file_for(self, base: str) -> str:
        """Map a logical file name like 'main' to a real filename for this lang."""
        if "." in base:
            return base
        return base + self.file_extension


_PROFILES: dict[Language, LanguageProfile] = {
    Language.PYTHON: LanguageProfile(
        name=Language.PYTHON,
        display="Python",
        file_extension=".py",
        entrypoint="main.py",
        test_extension=".py",
        test_framework="pytest",
        comment_style="#",
        package_manifest="requirements.txt",
        syntax_check_marker="```python",
        developer_system_addendum=(
            "Write clean, idiomatic Python. Use type hints. Use dataclasses "
            "for simple data models. Do NOT call input(); the code must run "
            "without user interaction. For GUI code, define classes/functions "
            "but never call mainloop()."
        ),
        test_system_addendum=(
            "Generate pytest tests. Use pytest fixtures where appropriate. "
            "Cover happy path, edge cases, and error conditions."
        ),
    ),
    Language.TYPESCRIPT: LanguageProfile(
        name=Language.TYPESCRIPT,
        display="TypeScript",
        file_extension=".ts",
        entrypoint="main.ts",
        test_extension=".test.ts",
        test_framework="vitest",
        comment_style="//",
        package_manifest="package.json",
        syntax_check_marker="```typescript",
        developer_system_addendum=(
            "Write clean, idiomatic TypeScript with strict types. Prefer "
            "interfaces over type aliases for object shapes. Use ES2022+ "
            "syntax. Do NOT use `any` unless absolutely necessary. The code "
            "must run under Node without user interaction."
        ),
        test_system_addendum=(
            "Generate Vitest tests. Use describe/it blocks and expect "
            "assertions. Cover happy path, edge cases, and error conditions."
        ),
    ),
}


def get_profile(language: Language | str | None = None) -> LanguageProfile:
    """Return the profile for ``language``, or for the active env default.

    Resolution order:
      1. Explicit ``language`` argument
      2. ``LANGUAGE`` env var
      3. Python
    """
    if language is None:
        env = os.environ.get("LANGUAGE", "").strip().lower()
        language = Language(env) if env else Language.PYTHON
    elif isinstance(language, str):
        language = Language(language)
    return _PROFILES[language]
