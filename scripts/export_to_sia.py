#!/usr/bin/env python3
"""Export captured agent interactions for fine-tuning via self-improving-agent.

Without the ``[sia]`` extra installed: writes per-role JSONL files in SIA's
expected schema and prints the SIA training command to run by hand.

With the ``[sia]`` extra installed: optionally chains into SIA's distillation
or fine-tuning pipeline directly via ``--run TARGET``.

Usage:
    pip install -e '.[sia]'   # one-time

    # Export only (no SIA install required)
    python scripts/export_to_sia.py [--role ROLE] [--out PATH] [--log-path PATH]

    # Export + drive SIA's distillation pipeline on the developer rows
    python scripts/export_to_sia.py --role developer --run distill

Defaults:
    --role     all   (exports every role to ./training_data/<role>.jsonl)
    --out      training_data/
    --log-path $INTERACTION_LOG_PATH or logs/agent_interactions.jsonl
    --run      none  (just export; don't call into SIA)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root on the path so `from core...` works when invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.interaction_log import enable_logging, export_for_sia

ALL_ROLES = [
    "planner", "architect", "developer", "critic",
    "qa", "integrator", "test_generator", "documenter", "researcher",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", default="all", help="Role to export, or 'all'.")
    parser.add_argument("--out", default="training_data", help="Output directory.")
    parser.add_argument(
        "--log-path",
        default=os.environ.get("INTERACTION_LOG_PATH", "logs/agent_interactions.jsonl"),
        help="Path to the interaction log.",
    )
    parser.add_argument(
        "--run",
        choices=["none", "distill", "validate"],
        default="none",
        help=(
            "After exporting, optionally drive SIA: "
            "'distill' runs SIA's distillation pipeline on the exported rows; "
            "'validate' just imports SIA's modules to confirm the install works."
        ),
    )
    args = parser.parse_args()

    enable_logging(args.log_path)
    if not Path(args.log_path).exists():
        print(f"No interaction log at {args.log_path}.")
        print("Run the pipeline with INTERACTION_LOG_PATH set first.")
        return 1

    roles = ALL_ROLES if args.role == "all" else [args.role]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    written: dict[str, Path] = {}
    for role in roles:
        out_path = out_dir / f"{role}.jsonl"
        count = export_for_sia(role, out_path)
        print(f"  {role:<16} -> {out_path}  ({count} row(s))")
        total += count
        if count > 0:
            written[role] = out_path

    print(f"\nWrote {total} row(s) across {len(roles)} role(s).")

    if args.run == "none":
        print("Feed each file to self-improving-agent's training pipeline:")
        print(
            f"  python -m self_improving_agent.training.finetune "
            f"--data {out_dir}/<role>.jsonl"
        )
        return 0

    # --run validate or --run distill: needs the [sia] extra
    if not _sia_available():
        print(
            "\nSIA isn't installed. Install with:\n"
            "    pip install -e '.[sia]'\n"
            "Then re-run with --run distill (or --run validate)."
        )
        return 2

    if args.run == "validate":
        return _run_validate()

    if args.run == "distill":
        return _run_distill(written)

    return 0


def _sia_available() -> bool:
    try:
        import self_improving_agent  # noqa: F401
        return True
    except ImportError:
        return False


def _run_validate() -> int:
    """Quick proof-of-life: import the SIA modules we'd actually use."""
    from self_improving_agent.evaluation.harness import EvaluationHarness  # noqa: F401
    from self_improving_agent.observability.cost_tracker import (  # noqa: F401
        begin_run,
        render_summary,
    )
    from self_improving_agent.reproducibility import set_seed  # noqa: F401
    from self_improving_agent.training.distillation import (  # noqa: F401
        ListQuestionPool,
        run_distillation,
    )
    from self_improving_agent.training.judges import (  # noqa: F401
        EnsembleJudge,
        ProviderJudge,
    )

    print(
        "\nSIA validation: every cross-repo module imported cleanly.\n"
        "  - self_improving_agent.evaluation.harness.EvaluationHarness\n"
        "  - self_improving_agent.observability.cost_tracker (begin_run/render_summary)\n"
        "  - self_improving_agent.reproducibility.set_seed\n"
        "  - self_improving_agent.training.distillation (run_distillation)\n"
        "  - self_improving_agent.training.judges (EnsembleJudge / ProviderJudge)\n"
    )
    return 0


def _run_distill(written: dict[str, Path]) -> int:
    """Drive SIA's distillation pipeline on the exported developer rows.

    Distillation takes (user_message -> better_response) pairs and trains
    a smaller model to mimic them. The developer role is the most useful
    starting point because its outputs are concrete code we can grade.
    """
    import asyncio
    import json

    from self_improving_agent.observability.cost_tracker import (
        begin_run,
        render_summary,
    )
    from self_improving_agent.reproducibility import set_seed
    from self_improving_agent.training.distillation import (
        ListQuestionPool,
        run_distillation,
    )

    set_seed()
    begin_run()

    target_role = "developer"
    if target_role not in written:
        print(
            f"No '{target_role}' rows exported — nothing for distillation. "
            "Re-run with --role developer (or --role all) first."
        )
        return 1

    # Pull the user messages out of the exported JSONL — those are our prompts
    prompts: list[str] = []
    with open(written[target_role]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for msg in row.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "").strip()
                    if content:
                        prompts.append(content)
                    break

    if not prompts:
        print(f"No user prompts found in {written[target_role]}.")
        return 1

    pool = ListQuestionPool(prompts=prompts[:20])  # cap during proof-of-concept
    out_path = Path("training_data") / "distilled_from_developer.jsonl"
    print(f"\nRunning SIA distillation on {len(pool.prompts)} prompt(s)...")
    try:
        report = asyncio.run(
            run_distillation(
                pool=pool,
                output_path=out_path,
                judge_threshold=7.0,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Distillation run failed: {exc}")
        return 1

    print(report.render())
    print()
    print(render_summary())
    print(f"\nDistilled pairs at: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
