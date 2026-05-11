#!/usr/bin/env python3
"""Export captured agent interactions for fine-tuning via self-improving-agent.

Usage:
    python scripts/export_to_sia.py [--role ROLE] [--out PATH] [--log-path PATH]

Defaults:
    --role     all   (exports every role to ./training_data/<role>.jsonl)
    --out      training_data/
    --log-path $INTERACTION_LOG_PATH or logs/agent_interactions.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root on the path so `from core...` works when invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.interaction_log import enable_logging, export_for_sia, get_log


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
    for role in roles:
        out_path = out_dir / f"{role}.jsonl"
        count = export_for_sia(role, out_path)
        print(f"  {role:<16} -> {out_path}  ({count} row(s))")
        total += count

    print(f"\nWrote {total} row(s) across {len(roles)} role(s).")
    print(f"Feed each file to self-improving-agent's training pipeline:")
    print(f"  python -m self_improving_agent.training.finetune --data {out_dir}/<role>.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
