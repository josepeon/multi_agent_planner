# multi_agent_planner/main.py

from agents.planner import PlannerAgent
from agents.developer import DeveloperAgent
from agents.qa import QAAgent
from agents.critic import CriticAgent

def main():
    planner = PlannerAgent()
    developer = DeveloperAgent()
    qa = QAAgent()
    critic = CriticAgent()

    print("\nUSER PROMPT")
    user_prompt = input("What would you like the system to build?\n> ")

    print("\nPLANNING STAGE")
    task_list = planner.plan_task(user_prompt)
    print("Subtasks:")
    for i, task in enumerate(task_list):
        print(f"  {i+1}. {task}")

    print("\nDEVELOPMENT + QA STAGE")
    for i, task in enumerate(task_list):
        print(f"\nTask {i+1}: {task}")
        max_retries = 2
        attempt = 0
        success = False
        feedback = None

        while attempt <= max_retries and not success:
            print(f"\nAttempt {attempt + 1}")
            code = developer.write_code(task, feedback)
            print("Generated Code:\n", code)

            evaluation = qa.evaluate_code(code)
            success = evaluation["success"]

            if success:
                print("✅ QA Passed")
            else:
                print("❌ QA Failed")
                print("Error:\n", evaluation["error"])

                critique = critic.critique_code(task, code, evaluation["error"])
                print("Critique:\n", critique)

                feedback = f"{evaluation['error']}\n\nCritique:\n{critique}"
                attempt += 1

if __name__ == "__main__":
    main()