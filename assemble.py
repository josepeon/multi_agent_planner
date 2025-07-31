"""
assemble.py

This script loads the task log JSON file and passes it to the code assembler to generate final code output.
"""

import json
from core.assembler import assemble_code_from_log

def main():
    try:
        with open("output/task_log.json") as f:
            log_data = json.load(f)
        assemble_code_from_log(log_data)
    except FileNotFoundError:
        print("Error: 'output/task_log.json' file not found.")
    except json.JSONDecodeError:
        print("Error: Failed to parse 'output/task_log.json'. Ensure it is valid JSON.")

if __name__ == "__main__":
    main()