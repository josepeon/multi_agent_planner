import os
import json
from datetime import datetime
from agents.planner import PlannerAgent
from agents.developer import DeveloperAgent
from agents.qa import QAAgent
from agents.critic import CriticAgent
from core.orchestrator import run_orchestrator
from core.memory import Memory
from core.task_schema import Task
from dataclasses import asdict

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

        task = Task(id=memory.get("last_task_id", 0) + 1, description=user_prompt)
        memory.set("last_task_id", int(task.id))

        result = run_orchestrator(task)
        task.status = "complete"
        task.result = result

        memory.set(f"task_{task.id}", asdict(task))
        save_log(asdict(task))

        print("\nFINAL OUTPUT:")
        print(result)

        again = input("\nWould you like to build another project? (y/n): ")
        if again.strip().lower() != 'y':
            print("Exiting the system. Goodbye!")
            break

if __name__ == "__main__":
    main()