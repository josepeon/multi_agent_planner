# multi_agent_planner/main.py

from agents.planner import PlannerAgent
from agents.developer import DeveloperAgent

def main():
    planner = PlannerAgent()
    developer = DeveloperAgent()

    print("\nUSER PROMPT")
    user_prompt = input("What would you like the system to build?\n> ")

    print("\nPLANNING STAGE")
    task_list = planner.plan_task(user_prompt)
    print("Subtasks:")
    for i, task in enumerate(task_list):
        print(f"  {i+1}. {task}")

    print("\nDEVELOPMENT STAGE")
    for i, task in enumerate(task_list):
        print(f"\nTask {i+1}: {task}")
        code = developer.write_code(task)
        print("Generated Code:")
        print(code)

if __name__ == "__main__":
    main()