"""
Orchestrator Module

Main pipeline that coordinates all agents:
Planner → Architect → Developer (with retry) → QA → Critic → Integrator → TestGenerator → Documenter

Supports parallel execution of independent tasks using ThreadPoolExecutor.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agents.architect import ArchitectAgent
from agents.critic import CriticAgent
from agents.developer import DeveloperAgent
from agents.documenter import DocumenterAgent
from agents.integrator import IntegratorAgent
from agents.planner import PlannerAgent
from agents.qa import QAAgent
from agents.test_generator import TestGeneratorAgent
from core import cost_tracker
from core.checkpoints import (
    ApproveDecision,
    AutoApproveHandler,
    CheckpointHandler,
    CheckpointPrompt,
    EditDecision,
    RegenerateDecision,
    get_handler_from_env,
    render_architecture,
    render_plan,
)
from core.events import emit as emit_event, get_bus
from core.memory_store import ProjectMemory, project_id_for
from core.cost_tracker import attribute, begin_run
from core.memory import Memory
from core.shared_context import SharedContext, get_shared_context, reset_shared_context
from core.task_schema import Task
from core.test_runner import render_summary, run_generated_tests, write_result_log


# Max times we re-run an agent on a single Regenerate decision before giving up
# and accepting whatever was produced. Prevents an infinite loop if the user
# keeps hitting regenerate forever.
MAX_CHECKPOINT_REGEN_TRIES = 3


def _run_checkpoint(
    handler: CheckpointHandler,
    stage: str,
    artifact,
    rendered: str,
    context: dict | None = None,
) -> tuple[object, str]:
    """Surface ``artifact`` to ``handler`` and return (artifact_or_replacement, status).

    ``status`` is one of:
      - "approved": handler accepted as-is
      - "edited":   handler returned a replacement; first element is the new artifact
      - "regenerate": handler asked the producer to retry; first element is a hint (str or None)
    """
    prompt = CheckpointPrompt(
        stage=stage,
        artifact=artifact,
        rendered=rendered,
        context=context or {},
    )
    decision = handler.handle(prompt)
    if isinstance(decision, ApproveDecision):
        return artifact, "approved"
    if isinstance(decision, EditDecision):
        return decision.new_artifact, "edited"
    if isinstance(decision, RegenerateDecision):
        return decision.hint, "regenerate"
    return artifact, "approved"

# Initialize agents
planner: PlannerAgent = PlannerAgent()
architect: ArchitectAgent = ArchitectAgent()
developer: DeveloperAgent = DeveloperAgent()
qa_checker: QAAgent = QAAgent()
critic: CriticAgent = CriticAgent()
integrator: IntegratorAgent = IntegratorAgent()
test_generator: TestGeneratorAgent = TestGeneratorAgent()
documenter: DocumenterAgent = DocumenterAgent()
memory: Memory = Memory(filepath="output/memory.json")
shared_context: SharedContext = get_shared_context()

# Configuration
MAX_RETRIES: int = 3  # Number of times to retry failed tasks with critic feedback
MULTI_FILE_OUTPUT: bool = True  # Whether to generate multi-file project structure


def develop_with_retry(task: Task, max_retries: int = MAX_RETRIES) -> dict[str, Any]:
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
        with attribute("developer"):
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
        with attribute("qa"):
            qa_result = qa_checker.evaluate_code(code)
        passed = qa_result.get('status') == 'passed'

        print(f"  QA Result: {'PASSED' if passed else 'FAILED'}")

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
            with attribute("critic"):
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


def run_pipeline(
    task: Task,
    save_path: str = "output/session_log.json",
    *,
    use_dag: bool | None = None,
    checkpoint_handler: CheckpointHandler | None = None,
    job_id: str | None = None,
) -> str:
    """
    Run the full multi-agent code generation pipeline.

    Args:
        task: The Task object containing the user's request
        save_path: Path to save the session log JSON
        use_dag: If True, dispatch to the DAG-based orchestrator
            (parallel module development, mid-run replanning).
            If None, honor the USE_DAG_PIPELINE env var; default False.
        checkpoint_handler: Optional human-in-the-loop handler. If None,
            honors CHECKPOINT_MODE env (default: auto-approve, equivalent to
            pre-checkpoint behavior).

    Returns:
        The final generated code as a string
    """
    if checkpoint_handler is None:
        checkpoint_handler = get_handler_from_env()

    if use_dag is None:
        use_dag = os.environ.get("USE_DAG_PIPELINE", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
    if use_dag:
        from core.orchestrator_dag import run_pipeline_dag

        prompt = task if isinstance(task, str) else task.description
        result = run_pipeline_dag(
            prompt,
            save_path=save_path,
            checkpoint_handler=checkpoint_handler,
            job_id=job_id,
        )
        return result["final_code"]

    # Linear pipeline below. job_id defaults to "_local" if not provided —
    # events still emit, just nobody is listening.
    if job_id is None:
        job_id = "_local"

    # Reset shared context and cost tracker for new session
    reset_shared_context()
    global shared_context
    shared_context = get_shared_context()
    begin_run()  # zero out per-run token/cost counters

    memory.set("last_prompt", task.description)

    # Store the original prompt before task variable gets reassigned
    original_prompt = task.description

    # Per-project memory keyed by stable hash of the prompt. Architects can
    # already read user-level memory; project memory persists decisions from
    # this run for future runs that revisit the same project.
    project_mem = ProjectMemory(project_id=project_id_for(original_prompt))

    run_start_ts = __import__("time").time()
    emit_event(job_id, "run_started", {"prompt": original_prompt, "mode": "linear"})
    emit_event(job_id, "stage_started", {"stage": "planner", "agent": "PlannerAgent"})

    # Planner stage (with optional human checkpoint)
    plan_hint = ""
    for regen in range(MAX_CHECKPOINT_REGEN_TRIES + 1):
        with attribute("planner"):
            tasks = planner.plan(
                original_prompt + (f"\n\nUser hint: {plan_hint}" if plan_hint else "")
            )

        artifact, status = _run_checkpoint(
            checkpoint_handler,
            "plan",
            tasks,
            render_plan(tasks),
            context={"prompt": original_prompt},
        )
        if status == "approved":
            break
        if status == "edited":
            tasks = artifact  # user replaced the plan
            break
        if status == "regenerate":
            plan_hint = artifact or ""
            if regen >= MAX_CHECKPOINT_REGEN_TRIES:
                print("  Reached max regenerate attempts on plan; accepting last output.")
                break
            continue

    memory.set("last_tasks", tasks)
    emit_event(job_id, "stage_finished", {
        "stage": "planner",
        "agent": "PlannerAgent",
        "ok": True,
        "summary": f"{len(tasks)} module(s)",
        "tasks": [t.description if isinstance(t, Task) else str(t) for t in tasks],
    })

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
    emit_event(job_id, "stage_started", {"stage": "architect", "agent": "ArchitectAgent"})
    task_descriptions = [t.description if isinstance(t, Task) else t for t in tasks]
    arch_hint = ""
    for regen in range(MAX_CHECKPOINT_REGEN_TRIES + 1):
        with attribute("architect"):
            architecture = architect.design(
                original_prompt
                + (f"\n\nUser hint: {arch_hint}" if arch_hint else ""),
                task_descriptions,
            )

        artifact, status = _run_checkpoint(
            checkpoint_handler,
            "architect",
            architecture,
            render_architecture(architecture),
            context={"prompt": original_prompt},
        )
        if status == "approved":
            break
        if status == "edited":
            architecture = artifact  # user-edited replacement
            break
        if status == "regenerate":
            arch_hint = artifact or ""
            if regen >= MAX_CHECKPOINT_REGEN_TRIES:
                print("  Reached max regenerate attempts on architecture; accepting last output.")
                break
            continue

    print(architect.get_design_summary() if hasattr(architect, "get_design_summary") else str(architecture))
    emit_event(job_id, "stage_finished", {
        "stage": "architect",
        "agent": "ArchitectAgent",
        "ok": True,
        "summary": str(getattr(architecture, "description", architecture))[:200],
    })
    # Record the architecture decision for future runs against this prompt
    arch_text = str(getattr(architecture, "description", architecture))
    if arch_text:
        project_mem.remember(arch_text[:1000], kind="architecture", tags=["v1"])

    session_log = {"prompt": original_prompt, "architecture": architecture.description, "tasks": []}

    passed_count = 0
    failed_count = 0

    for task in tasks:
        print(f"\n{'='*60}")
        print("DEVELOPMENT + QA STAGE")
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
            print(f"\n[OK] Task {task.id} PASSED (after {result['attempts']} attempt(s))")
            emit_event(job_id, "task_passed", {
                "task_id": task.id,
                "description": task.description[:80],
                "attempts": result["attempts"],
            })
            # Persist the passing decision for future runs against this prompt
            project_mem.remember(
                f"Module passing on attempt {result['attempts']}: {task.description}",
                kind="decision",
                tags=[f"task_{task.id}"],
            )
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
            print(f"\n[FAIL] Task {task.id} FAILED (after {result['attempts']} attempts)")
            emit_event(job_id, "task_failed", {
                "task_id": task.id,
                "description": task.description[:80],
                "attempts": result["attempts"],
            })
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
    print(f"\nSession log saved to: {save_path}")

    # Integrate final code using the Integrator agent
    print(f"\n{'='*60}")
    print("INTEGRATION STAGE")
    print(f"{'='*60}")

    # Choose single or multi-file output
    with attribute("integrator"):
        if MULTI_FILE_OUTPUT:
            print("  Mode: Multi-file project structure")
            integrator.integrate_multifile(session_log, "output/project")

            # Also create a combined final_program.py for backwards compatibility
            final_code = integrator.integrate(session_log)
            with open("output/final_program.py", "w") as f:
                f.write(final_code)
            print("  Single-file backup: output/final_program.py")
        else:
            final_code = integrator.integrate(session_log)
            with open("output/final_program.py", "w") as f:
                f.write(final_code)
            print("\nFinal program saved to: output/final_program.py")

    # Run test generation and documentation in parallel
    print(f"\n{'='*60}")
    print("TEST + DOCUMENTATION STAGE (Parallel)")
    print(f"{'='*60}")

    def generate_tests_task():
        """Generate pytest tests for the final code."""
        with attribute("test_generator"):
            return test_generator.generate_tests(final_code)

    def generate_docs_task():
        """Generate README documentation."""
        with attribute("documenter"):
            return documenter.generate_readme(final_code, original_prompt)

    # Execute both tasks in parallel
    test_code = None
    readme = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(generate_tests_task): "tests",
            executor.submit(generate_docs_task): "docs"
        }

        for future in as_completed(futures):
            task_name = futures[future]
            try:
                result = future.result()
                if task_name == "tests":
                    test_code = result
                    print("  Test generation complete")
                else:
                    readme = result
                    print("  Documentation generation complete")
            except Exception as e:
                print(f"  {task_name} failed: {e}")
                if task_name == "tests":
                    test_code = "# Test generation failed"
                else:
                    readme = "# Documentation generation failed"

    # Save outputs
    with open("output/test_program.py", "w") as f:
        f.write(test_code)
    print("Tests saved to: output/test_program.py")

    with open("output/README.md", "w") as f:
        f.write(readme)
    print("README saved to: output/README.md")
    memory.set("last_tests", test_code)
    memory.set("last_readme", readme)

    # Verification stage: actually run the generated tests against the
    # generated program. Without this step, "tests pass" is unverified.
    print(f"\n{'='*60}")
    print("TEST EXECUTION STAGE")
    print(f"{'='*60}")
    test_result = run_generated_tests(final_code, test_code)
    print(render_summary(test_result))
    write_result_log(test_result, "output/test_results.json")
    session_log["test_run"] = test_result.as_dict()

    # Per-run cost & token usage report
    print(f"\n{'='*60}")
    print("COST & TOKEN REPORT")
    print(f"{'='*60}")
    print(cost_tracker.render_summary())
    cost_snapshot = cost_tracker.to_dict()
    session_log["cost"] = cost_snapshot
    with open("output/cost_report.json", "w") as f:
        json.dump(cost_snapshot, f, indent=2)
    memory.set("last_cost_report", cost_snapshot)

    with open(save_path, "w") as f:
        json.dump(session_log, f, indent=2)
    memory.set("last_test_run", test_result.as_dict())

    emit_event(job_id, "test_run", test_result.as_dict())
    emit_event(job_id, "cost", cost_snapshot)
    emit_event(job_id, "run_finished", {
        "duration_s": __import__("time").time() - run_start_ts,
        "ok": test_result.all_passed,
    })
    get_bus().end(job_id)

    return final_code

# Export run_pipeline as run_orchestrator for import
run_orchestrator = run_pipeline

if __name__ == "__main__":
    user_input = input("Enter your project description: ")
    run_pipeline(user_input)
