import json
from dataclasses import asdict
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

def run_pipeline(task: Task, save_path="output/session_log.json"):

    memory.set("last_prompt", task.description)

    tasks = planner.plan(task.description)
    memory.set("last_tasks", tasks)

    print("PLANNING STAGE")
    for idx, task in enumerate(tasks, 1):
        description = task.description if isinstance(task, Task) else task
        clean_description = description.strip().lstrip('*').strip()
        print(f"  Task {idx}: {clean_description}")
        if not isinstance(task, Task):
            task = Task(id=idx - 1, description=description)
            tasks[idx - 1] = task

    session_log = {"prompt": task.description, "tasks": []}

    for task in tasks:
        print(f"\nDEVELOPMENT + QA STAGE\n\nTask {task.id}: {task.description} (Status: {task.status})")

        # Develop code
        code = developer.develop(task)
        print(f"\nGenerated Code for Task {task.id}:\n{code}\n")

        # Run QA
        qa_result = qa_checker.evaluate_code(code)
        print(f"\nQA Result for Task {task.id}: {'✅ Passed' if qa_result.get('status') == 'passed' else '❌ Failed'}")

        # Optionally call critic
        critique = ""
        if qa_result.get('status') != "passed":
            critique = critic.review(task.description, code, qa_result.get('result'))
            print("\nCritique:\n", critique)

        # Append to session log
        task.status = "complete" if qa_result.get('status') == "passed" else "failed"
        task.result = code

        session_log["tasks"].append({
            "task": task.description,
            "code": code.get("code"),
            "result": code.get("result"),
            "status": code.get("status"),
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
