import os
import re
from typing import List, Dict, Any
from core.memory import Memory


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
        task.get("code", "")
        for task in log_data.get("tasks", [])
        if task.get("qa_result", {}).get("success") and task.get("code", "").strip()
    ]

    final_code = extract_and_clean_code(successful_code_blocks)

    memory = Memory(filepath="output/memory.json")
    memory.set("final_code", final_code)

    save_final_script(final_code)
    return final_code