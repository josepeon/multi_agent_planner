import json
from agents.planner import PlannerAgent
from agents.developer import DeveloperAgent
from agents.qa import QAChecker
from agents.critic import CriticAgent
from core.assembler import assemble_code_from_log

# Initialize agents
planner = PlannerAgent()
developer = DeveloperAgent()
qa_checker = QAChecker()
critic = CriticAgent()

def run_pipeline(user_prompt, save_path="output/session_log.json"):
    print("\nUSER PROMPT")
    print(f"What would you like the system to build?\n> {user_prompt}\n")

    tasks = planner.plan(user_prompt)
    print("PLANNING STAGE")
    for idx, task in enumerate(tasks, 1):
        print(f"  {idx}. {task}")

    session_log = {"prompt": user_prompt, "tasks": []}

    for task in tasks:
        print(f"\nDEVELOPMENT + QA STAGE\n\nTask: {task}")

        # Develop code
        code = developer.develop(task)
        print("\nGenerated Code:\n", code)

        # Run QA
        qa_result = qa_checker.evaluate_code(code)
        print("\nQA Result:", "✅ Passed" if qa_result.success else "❌ Failed")

        # Optionally call critic
        critique = ""
        if not qa_result.success:
            critique = critic.critique(code, qa_result.error)
            print("\nCritique:\n", critique)

        # Append to session log
        session_log["tasks"].append({
            "task": task,
            "code": code,
            "qa_result": qa_result.to_dict(),
            "critique": critique,
        })

    # Save log
    with open(save_path, "w") as f:
        json.dump(session_log, f, indent=2)
    print(f"\n📝 Session log saved to: {save_path}")

    # Assemble final code
    print("\nASSEMBLING FINAL PROGRAM")
    final_code = assemble_code_from_log(session_log)
    return final_code

if __name__ == "__main__":
    user_input = input("Enter your project description: ")
    run_pipeline(user_input)
