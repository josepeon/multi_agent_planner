import json
from agents.planner import PlannerAgent
from agents.developer import DeveloperAgent
from agents.qa import QAAgent
from agents.critic import CriticAgent
from core.assembler import assemble_code_from_log
from core.memory import Memory
from core.task_schema import Task

# Initialize agents
planner = PlannerAgent()
developer = DeveloperAgent()
qa_checker = QAAgent()
critic = CriticAgent()
memory = Memory(filepath="output/memory.json")

def run_pipeline(user_prompt, save_path="output/session_log.json"):
    print("\nUSER PROMPT")
    print(f"What would you like the system to build?\n> {user_prompt}\n")

    memory.set("last_prompt", user_prompt)

    tasks = planner.plan(user_prompt)
    memory.set("last_tasks", tasks)

    print("PLANNING STAGE")
    for idx, description in enumerate(tasks, 1):
        task = Task(id=idx, description=description)
        print(f"  {task.id}. {task.description}")
        tasks[idx - 1] = task

    session_log = {"prompt": user_prompt, "tasks": []}

    for task in tasks:
        print(f"\nDEVELOPMENT + QA STAGE\n\nTask: {task.description}")

        # Develop code
        code = developer.develop(task.description)
        print("\nGenerated Code:\n", code)

        # Run QA
        qa_result = qa_checker.evaluate_code(code)
        print("\nQA Result:", "✅ Passed" if qa_result.get("success") else "❌ Failed")

        # Optionally call critic
        critique = ""
        if not qa_result.get("success"):
            critique = critic.review(task.description, code, qa_result.get("error"))
            print("\nCritique:\n", critique)

        # Append to session log
        task.status = "complete" if qa_result.get("success") else "failed"
        task.result = code

        session_log["tasks"].append({
            "task": task.description,
            "code": code,
            "qa_result": qa_result,
            "critique": critique,
        })

    # Save log
    with open(save_path, "w") as f:
        json.dump(session_log, f, indent=2)
    print(f"\n📝 Session log saved to: {save_path}")

    # Assemble final code
    print("\nASSEMBLING FINAL PROGRAM")
    final_code = assemble_code_from_log(session_log)
    memory.set("last_final_code", final_code)
    return final_code

# Export run_pipeline as run_orchestrator for import
run_orchestrator = run_pipeline

if __name__ == "__main__":
    user_input = input("Enter your project description: ")
    run_pipeline(user_input)
