import os
import re
from typing import List, Dict, Any
from core.memory import Memory
from core.task_schema import Task


def extract_and_clean_code(code_blocks: List[str]) -> str:
    """
    Combine code blocks, remove duplicate imports, organize into a single file.
    """
    all_code = "\n\n".join(code_blocks)

    # Extract and deduplicate import statements
    import_lines = set(re.findall(r"^import\s.+?$|^from\s.+?\simport\s.+?$", all_code, re.MULTILINE))
    code_wo_imports = re.sub(r"^import\s.+?$|^from\s.+?\simport\s.+?$", "", all_code, flags=re.MULTILINE)

    cleaned_code = "\n".join(sorted(import_lines)) + "\n\n" + code_wo_imports.strip()
    return cleaned_code


def save_final_script(code: str, path: str = "output/final_program.py") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(code)
    print(f"\n✅ Final program saved to: {path}")


def assemble_code_from_log(log_data: Dict[str, Any]) -> str:
    successful_code_blocks = [
        t["code"]
        for t in log_data.get("tasks", [])
        if (
            t.get("result")
            and isinstance(t.get("result"), str)
            and t["result"].strip()
            and isinstance(t.get("code"), str)
            and not t["code"].strip().startswith("# Create a file")
        )
    ]

    # Deduplicate code blocks
    seen = set()
    unique_blocks = []
    for block in successful_code_blocks:
        if block not in seen:
            unique_blocks.append(block)
            seen.add(block)

    final_code = extract_and_clean_code(unique_blocks)

    memory = Memory(filepath="output/memory.json")
    memory.set("final_code", final_code)

    save_final_script(final_code)
    return final_code