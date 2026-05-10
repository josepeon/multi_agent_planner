"""
DAG-based pipeline orchestrator.

Wraps the existing agents in ``PipelineNode`` adapters so the agent pipeline
becomes a dependency graph with parallel module development. Coexists with
``core.orchestrator.run_pipeline`` (the linear version) — the new entry point
is ``run_pipeline_dag(prompt)`` and the linear orchestrator forwards to it
when ``USE_DAG_PIPELINE=1`` (env) or ``run_pipeline(..., use_dag=True)``.

Graph shape for a project with N planner-emitted module tasks:

    plan ──► architect ──► dev_0 ──► qa_0 ──┐
                       ╲              ...   │
                       ╲   ──► dev_(N-1) ──► qa_(N-1) ──┐
                                                        ▼
                                                   integrate
                                                  /         \\
                                                 ▼           ▼
                                              tests        docs
                                                 \\        /
                                                  ▼       ▼
                                                test_runner

dev_i runs in parallel — that's the win over the linear version. qa_i still
follows its dev_i. Tests + docs run in parallel (preserved from the linear
orchestrator). The test_runner node ties everything together.
"""

from __future__ import annotations

import json
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
from core.cost_tracker import attribute, begin_run
from core.memory import Memory
from core.pipeline_graph import (
    NodeResult,
    PipelineGraph,
    PipelineNode,
    SKIPPED,
)
from core.shared_context import SharedContext, get_shared_context, reset_shared_context
from core.task_schema import Task
from core.test_runner import render_summary, run_generated_tests, write_result_log


MAX_CHECKPOINT_REGEN_TRIES = 3


def _apply_checkpoint(handler: CheckpointHandler, prompt: CheckpointPrompt):
    """Returns (artifact_or_hint, status). Same protocol as the linear orchestrator."""
    decision = handler.handle(prompt)
    if isinstance(decision, ApproveDecision):
        return prompt.artifact, "approved"
    if isinstance(decision, EditDecision):
        return decision.new_artifact, "edited"
    if isinstance(decision, RegenerateDecision):
        return decision.hint, "regenerate"
    return prompt.artifact, "approved"

MAX_DEV_RETRIES: int = 3
MULTI_FILE_OUTPUT: bool = True


def _develop_with_retry_node(
    task: Task,
    developer: DeveloperAgent,
    qa: QAAgent,
    critic: CriticAgent,
    shared_context: SharedContext,
) -> dict[str, Any]:
    """Same retry loop as the linear orchestrator, packaged as a node body."""
    feedback = None
    best_code: dict[str, Any] | None = None
    best_qa: dict[str, Any] | None = None
    critique = ""
    context_summary = shared_context.get_context_summary()

    for attempt in range(MAX_DEV_RETRIES):
        context_message = (
            f"\n\n## Context from previous tasks:\n{context_summary}"
            if context_summary
            else ""
        )
        with attribute("developer"):
            if feedback:
                code = developer.develop(task, feedback_message=feedback + context_message)
            elif (
                context_summary
                and context_summary != "No previous context. This is the first task."
            ):
                code = developer.develop(task, feedback_message=context_message)
            else:
                code = developer.develop(task)

        with attribute("qa"):
            qa_result = qa.evaluate_code(code)
        passed = qa_result.get("status") == "passed"

        if best_code is None or passed:
            best_code = code
            best_qa = qa_result

        if passed:
            return {
                "task": task,
                "code": code,
                "qa_result": qa_result,
                "critique": "",
                "attempts": attempt + 1,
                "passed": True,
            }

        if attempt < MAX_DEV_RETRIES - 1:
            with attribute("critic"):
                critique = critic.review(
                    task.description, code, qa_result.get("result")
                )
            feedback = (
                f"Previous attempt failed. Critic feedback:\n{critique}\n\nPlease fix these issues."
            )

    return {
        "task": task,
        "code": best_code,
        "qa_result": best_qa,
        "critique": critique,
        "attempts": MAX_DEV_RETRIES,
        "passed": False,
    }


def build_pipeline_graph(
    prompt: str,
    *,
    planner: PlannerAgent,
    architect: ArchitectAgent,
    developer: DeveloperAgent,
    qa: QAAgent,
    critic: CriticAgent,
    integrator: IntegratorAgent,
    test_generator: TestGeneratorAgent,
    documenter: DocumenterAgent,
    shared_context: SharedContext,
    checkpoint_handler: CheckpointHandler | None = None,
) -> PipelineGraph:
    """Construct the DAG.

    The plan node is a single root. After it runs, we *can't* know how many
    dev_* nodes to add until we have the planner's output. Two ways to handle
    that: (a) re-plan after the planner runs, or (b) make the architect node
    responsible for spawning dev nodes via Replan(). We use option (b) — the
    architect already inspects the task list, so it's the natural place.
    """
    g = PipelineGraph()
    handler = checkpoint_handler or AutoApproveHandler()

    def planner_run(_inputs: dict[str, Any]) -> list[Task]:
        hint = ""
        for regen in range(MAX_CHECKPOINT_REGEN_TRIES + 1):
            with attribute("planner"):
                tasks = planner.plan(
                    prompt + (f"\n\nUser hint: {hint}" if hint else "")
                )
            # Normalize to Task objects with stable ids
            out: list[Task] = []
            for i, t in enumerate(tasks):
                if isinstance(t, Task):
                    t.id = i
                    out.append(t)
                else:
                    out.append(Task(id=i, description=str(t)))

            artifact, status = _apply_checkpoint(
                handler,
                CheckpointPrompt(
                    stage="plan",
                    artifact=out,
                    rendered=render_plan(out),
                    context={"prompt": prompt},
                ),
            )
            if status == "approved":
                return out
            if status == "edited":
                return artifact  # user-replaced task list
            if status == "regenerate":
                hint = artifact or ""
                if regen >= MAX_CHECKPOINT_REGEN_TRIES:
                    return out
                continue
        return out

    g.add(PipelineNode(id="plan", run=planner_run, parallel=False, role="planner"))

    def architect_run(inputs: dict[str, Any]) -> dict[str, Any]:
        tasks: list[Task] = inputs["plan"]
        descriptions = [t.description for t in tasks]
        arch_hint = ""
        architecture = None
        for regen in range(MAX_CHECKPOINT_REGEN_TRIES + 1):
            with attribute("architect"):
                architecture = architect.design(
                    prompt + (f"\n\nUser hint: {arch_hint}" if arch_hint else ""),
                    descriptions,
                )
            artifact, status = _apply_checkpoint(
                handler,
                CheckpointPrompt(
                    stage="architect",
                    artifact=architecture,
                    rendered=render_architecture(architecture),
                    context={"prompt": prompt},
                ),
            )
            if status == "approved":
                break
            if status == "edited":
                architecture = artifact
                break
            if status == "regenerate":
                arch_hint = artifact or ""
                if regen >= MAX_CHECKPOINT_REGEN_TRIES:
                    break
                continue

        # Spawn one dev_<i> per task; each feeds qa+critic via the inline retry.
        # Done as a Replan so the executor schedules them in the next layer.
        new_nodes: list[PipelineNode] = []
        for task in tasks:
            new_nodes.append(
                PipelineNode(
                    id=f"dev_{task.id}",
                    run=lambda _inputs, task=task: _develop_with_retry_node(
                        task, developer, qa, critic, shared_context
                    ),
                    depends_on=["architect"],  # all devs depend on architect
                    parallel=True,
                    role="developer",
                )
            )

        # The integrate node depends on every dev_<i>.
        new_nodes.append(
            PipelineNode(
                id="integrate",
                run=lambda inputs, tasks=tasks: _integrate(inputs, tasks, integrator, prompt),
                depends_on=[f"dev_{t.id}" for t in tasks],
                parallel=False,
                role="integrator",
            )
        )

        # tests + docs from integrator output, in parallel
        new_nodes.append(
            PipelineNode(
                id="tests",
                run=lambda inputs: _generate_tests(inputs, test_generator),
                depends_on=["integrate"],
                parallel=True,
                role="test_generator",
            )
        )
        new_nodes.append(
            PipelineNode(
                id="docs",
                run=lambda inputs: _generate_docs(inputs, documenter, prompt),
                depends_on=["integrate"],
                parallel=True,
                role="documenter",
            )
        )

        # final test execution depends on both tests + integrate (we need the
        # final program file content, which is the integrator's output)
        new_nodes.append(
            PipelineNode(
                id="run_tests",
                run=_run_tests,
                depends_on=["integrate", "tests"],
                parallel=False,
                role="test_runner",
            )
        )

        from core.pipeline_graph import Replan
        return Replan(new_nodes=new_nodes)

    g.add(
        PipelineNode(
            id="architect",
            run=architect_run,
            depends_on=["plan"],
            parallel=False,
            role="architect",
        )
    )

    return g


def _integrate(
    inputs: dict[str, Any],
    tasks: list[Task],
    integrator: IntegratorAgent,
    prompt: str,
) -> dict[str, Any]:
    """Build a session_log from dev results, then integrate."""
    session_log: dict[str, Any] = {"prompt": prompt, "tasks": []}
    for task in tasks:
        dev = inputs.get(f"dev_{task.id}")
        if dev is SKIPPED or dev is None:
            session_log["tasks"].append({
                "task": task.description,
                "code": "",
                "result": "",
                "status": "failed",
                "qa_result": {},
                "critique": "",
                "attempts": 0,
            })
            continue
        code = dev["code"]
        code_str = code.get("code") if isinstance(code, dict) else code
        session_log["tasks"].append({
            "task": task.description,
            "code": code_str,
            "result": code.get("result") if isinstance(code, dict) else "",
            "status": "complete" if dev["passed"] else "failed",
            "qa_result": dev["qa_result"],
            "critique": dev["critique"],
            "attempts": dev["attempts"],
        })

    with attribute("integrator"):
        if MULTI_FILE_OUTPUT:
            integrator.integrate_multifile(session_log, "output/project")
        final_code = integrator.integrate(session_log)

    with open("output/final_program.py", "w") as f:
        f.write(final_code)

    return {"final_code": final_code, "session_log": session_log}


def _generate_tests(inputs: dict[str, Any], test_generator: TestGeneratorAgent) -> str:
    integrate_out = inputs["integrate"]
    if integrate_out is SKIPPED:
        return "# Test generation skipped (integration failed)"
    final_code = integrate_out["final_code"]
    with attribute("test_generator"):
        test_code = test_generator.generate_tests(final_code)
    with open("output/test_program.py", "w") as f:
        f.write(test_code)
    return test_code


def _generate_docs(
    inputs: dict[str, Any], documenter: DocumenterAgent, prompt: str
) -> str:
    integrate_out = inputs["integrate"]
    if integrate_out is SKIPPED:
        return "# Documentation skipped (integration failed)"
    final_code = integrate_out["final_code"]
    with attribute("documenter"):
        readme = documenter.generate_readme(final_code, prompt)
    with open("output/README.md", "w") as f:
        f.write(readme)
    return readme


def _run_tests(inputs: dict[str, Any]) -> dict[str, Any]:
    integrate_out = inputs["integrate"]
    test_code = inputs["tests"]
    if integrate_out is SKIPPED or test_code is SKIPPED:
        return {"ran": False, "skip_reason": "upstream stage failed"}

    final_code = integrate_out["final_code"]
    test_result = run_generated_tests(final_code, test_code)
    write_result_log(test_result, "output/test_results.json")
    return test_result.as_dict()


def run_pipeline_dag(
    prompt: str,
    *,
    save_path: str = "output/session_log.json",
    on_node_start=None,
    on_node_finish=None,
    checkpoint_handler: CheckpointHandler | None = None,
) -> dict[str, Any]:
    """Execute the DAG pipeline and write the same output artifacts as the
    linear orchestrator (final_program.py, test_program.py, README.md,
    test_results.json, cost_report.json, session_log.json).

    Returns a dict with keys: final_code, session_log, test_run, cost,
    graph_layers, graph_duration.
    """
    reset_shared_context()
    shared_context = get_shared_context()
    begin_run()

    memory = Memory("output/memory.json")
    memory.set("last_prompt", prompt)

    planner = PlannerAgent()
    architect = ArchitectAgent()
    developer = DeveloperAgent()
    qa = QAAgent()
    critic = CriticAgent()
    integrator = IntegratorAgent()
    test_generator = TestGeneratorAgent()
    documenter = DocumenterAgent()

    graph = build_pipeline_graph(
        prompt,
        planner=planner,
        architect=architect,
        developer=developer,
        qa=qa,
        critic=critic,
        integrator=integrator,
        test_generator=test_generator,
        documenter=documenter,
        shared_context=shared_context,
        checkpoint_handler=checkpoint_handler or get_handler_from_env(),
    )

    result = graph.execute(
        max_workers=4,
        on_node_start=on_node_start,
        on_node_finish=on_node_finish,
    )

    integrate_out = (
        result.output_of("integrate")
        if "integrate" in result.nodes and result.succeeded("integrate")
        else None
    )
    final_code = integrate_out["final_code"] if integrate_out else ""
    session_log = integrate_out["session_log"] if integrate_out else {"prompt": prompt}

    test_run = (
        result.output_of("run_tests")
        if "run_tests" in result.nodes and result.succeeded("run_tests")
        else {"ran": False, "skip_reason": "test_runner did not run"}
    )
    session_log["test_run"] = test_run

    print("\n" + "=" * 60)
    print("DAG SUMMARY")
    print("=" * 60)
    print(f"Layers executed: {result.layers_executed}")
    print(f"Total time: {result.total_duration_seconds:.2f}s")
    for nid, nr in result.nodes.items():
        print(f"  {nid:<20} {nr.status:<8} layer={nr.layer} {nr.duration_seconds:.2f}s")

    if test_run.get("ran"):
        # Synthesize a TestRunResult-like object for the existing renderer
        from core.test_runner import TestRunResult

        tr = TestRunResult(
            ran=test_run.get("ran", False),
            passed=test_run.get("passed", 0),
            failed=test_run.get("failed", 0),
            errors=test_run.get("errors", 0),
            skipped=test_run.get("skipped", 0),
            total=test_run.get("total", 0),
            duration_seconds=test_run.get("duration_seconds", 0.0),
            failure_summaries=test_run.get("failure_summaries", []),
            skip_reason=test_run.get("skip_reason"),
        )
        print("\n" + render_summary(tr))

    print("\n" + cost_tracker.render_summary())
    cost_snapshot = cost_tracker.to_dict()
    session_log["cost"] = cost_snapshot
    session_log["graph"] = {
        "layers": result.layers_executed,
        "duration_seconds": result.total_duration_seconds,
        "nodes": {
            nid: {
                "status": nr.status,
                "layer": nr.layer,
                "duration_seconds": nr.duration_seconds,
                "error": nr.error,
            }
            for nid, nr in result.nodes.items()
        },
    }

    with open(save_path, "w") as f:
        json.dump(session_log, f, indent=2, default=str)
    with open("output/cost_report.json", "w") as f:
        json.dump(cost_snapshot, f, indent=2)

    memory.set("last_test_run", test_run)
    memory.set("last_cost_report", cost_snapshot)

    return {
        "final_code": final_code,
        "session_log": session_log,
        "test_run": test_run,
        "cost": cost_snapshot,
        "graph_layers": result.layers_executed,
        "graph_duration": result.total_duration_seconds,
    }
