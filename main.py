import os
import json
from datetime import datetime
from agents.planner import PlannerAgent
from agents.developer import DeveloperAgent
from agents.qa import QAAgent
from agents.critic import CriticAgent
from core.orchestrator import run_orchestrator
from core.memory import Memory

def save_log(log_data, path="output/task_log.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log_data["timestamp"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(log_data, f, indent=2)

def main():
    memory = Memory(filepath="output/memory.json")
    while True:
        print("\nUSER PROMPT")
        user_prompt = input("What would you like the system to build?\n> ")

        result = run_orchestrator(user_prompt)
        memory.set("last_result", result)
        save_log(result)

        print("\nFINAL OUTPUT:")
        print(result)

        again = input("\nWould you like to build another project? (y/n): ")
        if again.strip().lower() != 'y':
            print("Exiting the system. Goodbye!")
            break

if __name__ == "__main__":
    main()