"""
Orchestrator Module

Main pipeline that coordinates all agents:
Planner → Architect → Developer (with retry) → QA → Critic → Integrator
"""

import json
from agents.planner import PlannerAgent
from agents.developer import DeveloperAgent
from agents.qa import QAAgent
from agents.critic import CriticAgent
from agents.architect import ArchitectAgent
from agents.integrator import IntegratorAgent
from core.memory import Memory
from core.task_schema import Task
from core.shared_context import get_shared_context, reset_shared_context

# Initialize agents
planner = PlannerAgent()
architect = ArchitectAgent()
developer = DeveloperAgent()
qa_checker = QAAgent()
critic = CriticAgent()
integrator = IntegratorAgent()
memory = Memory(filepath="output/memory.json")
shared_context = get_shared_context()

# Configuration
MAX_RETRIES = 3  # Number of times to retry failed tasks with critic feedback


def develop_with_retry(task: Task, max_retries: int = MAX_RETRIES) -> dict:
    """
    Develop code for a task with retry logic.
    If QA fails, get critic feedback and retry up to max_retries times.
    Uses shared context to provide awareness of already-defined code.
    """
    feedback = None
    best_code = None
    best_qa_result = None
    critique = ""
    
    # Get context summary for the developer
    context_summary = shared_context.get_context_summary()
    
    for attempt in range(max_retries):
        # Develop code (pass feedback and context from previous attempt if any)
        context_message = f"\n\n## Context from previous tasks:\n{context_summary}" if context_summary else ""
        if feedback:
            full_feedback = feedback + context_message
            code = developer.develop(task, feedback_message=full_feedback)
        else:
            if context_summary and context_summary != "No previous context. This is the first task.":
                code = developer.develop(task, feedback_message=context_message)
            else:
                code = developer.develop(task)
        
        print(f"\n  Generated Code (Attempt {attempt + 1}/{max_retries}):")
        print(f"  {code}\n")
        
        # Run QA
        qa_result = qa_checker.evaluate_code(code)
        passed = qa_result.get('status') == 'passed'
        
        print(f"  QA Result: {'✅ Passed' if passed else '❌ Failed'}")
        
        # Track best attempt
        if best_code is None or passed:
            best_code = code
            best_qa_result = qa_result
        
        if passed:
            return {
                "code": code,
                "qa_result": qa_result,
                "critique": "",
                "attempts": attempt + 1,
                "passed": True
            }
        
        # Get critic feedback for retry
        if attempt < max_retries - 1:  # Don't get feedback on last attempt
            critique = critic.review(task.description, code, qa_result.get('result'))
            feedback = f"Previous attempt failed. Critic feedback:\n{critique}\n\nPlease fix these issues."
            print(f"\n  Critic Feedback (will retry):\n  {critique[:200]}...")
    
    # Return best attempt after all retries exhausted
    return {
        "code": best_code,
        "qa_result": best_qa_result,
        "critique": critique,
        "attempts": max_retries,
        "passed": False
    }


def run_pipeline(task: Task, save_path="output/session_log.json"):
    # Reset shared context for new session
    reset_shared_context()
    global shared_context
    shared_context = get_shared_context()

    memory.set("last_prompt", task.description)

    # Store the original prompt before task variable gets reassigned
    original_prompt = task.description

    tasks = planner.plan(original_prompt)
    memory.set("last_tasks", tasks)

    print("PLANNING STAGE")
    for idx, t in enumerate(tasks, 1):
        description = t.description if isinstance(t, Task) else t
        clean_description = description.strip().lstrip('*').strip()
        print(f"  Task {idx}: {clean_description}")
        if not isinstance(t, Task):
            t = Task(id=idx - 1, description=description)
            tasks[idx - 1] = t

    # Architecture design phase
    print(f"\n{'='*60}")
    print("ARCHITECTURE STAGE")
    print(f"{'='*60}")
    task_descriptions = [t.description if isinstance(t, Task) else t for t in tasks]
    architecture = architect.design(original_prompt, task_descriptions)
    print(architect.get_design_summary())

    session_log = {"prompt": original_prompt, "architecture": architecture.description, "tasks": []}
    
    passed_count = 0
    failed_count = 0

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"DEVELOPMENT + QA STAGE")
        print(f"Task {task.id}: {task.description}")
        print(f"{'='*60}")

        # Develop with retry loop
        result = develop_with_retry(task, MAX_RETRIES)
        
        code = result["code"]
        qa_result = result["qa_result"]
        critique = result["critique"]
        
        # Extract the actual code string
        code_str = code.get("code") if isinstance(code, dict) else code
        
        if result["passed"]:
            passed_count += 1
            task.status = "complete"
            print(f"\n✅ Task {task.id} PASSED (after {result['attempts']} attempt(s))")
            # Store successful code in shared context for future tasks
            shared_context.add_generated_code(
                task_id=task.id,
                name=task.description[:50],
                code=code_str,
                status="passed"
            )
        else:
            failed_count += 1
            task.status = "failed"
            print(f"\n❌ Task {task.id} FAILED (after {result['attempts']} attempts)")
            if critique:
                print(f"\nFinal Critique:\n{critique}")
        
        task.result = code

        session_log["tasks"].append({
            "task": task.description,
            "code": code_str,
            "result": code.get("result") if isinstance(code, dict) else "",
            "status": code.get("status") if isinstance(code, dict) else task.status,
            "qa_result": qa_result,
            "critique": critique,
            "attempts": result["attempts"],
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed_count} passed, {failed_count} failed out of {len(tasks)} tasks")
    print(f"{'='*60}")

    # Save log
    with open(save_path, "w") as f:
        json.dump(session_log, f, indent=2)
    print(f"\n📝 Session log saved to: {save_path}")

    # Integrate final code using the Integrator agent
    print(f"\n{'='*60}")
    print("INTEGRATION STAGE")
    print(f"{'='*60}")
    final_code = integrator.integrate(session_log)
    
    # Save final code
    with open("output/final_program.py", "w") as f:
        f.write(final_code)
    print(f"\n✅ Final program saved to: output/final_program.py")
    
    memory.set("last_final_code", final_code)
    return final_code

# Export run_pipeline as run_orchestrator for import
run_orchestrator = run_pipeline

if __name__ == "__main__":
    user_input = input("Enter your project description: ")
    run_pipeline(user_input)
