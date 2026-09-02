#!/usr/bin/env python3
"""Re-evaluate saved baseline checkpoints to add detailed metrics without retraining."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_ten_motion_baseline import MOTIONS, PROJECT_ROOT, RESULT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", nargs="+", choices=MOTIONS, default=list(MOTIONS))
    parser.add_argument("--state_file", type=Path, default=RESULT_ROOT / "orchestrator_state.json")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=RESULT_ROOT / "reevaluation_v2",
        help="Independent output root; original evaluation artifacts are never overwritten.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite evaluations that already contain detailed data")
    parser.add_argument("--video", action="store_true", help="Also record a new video; existing videos are kept by default")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    state = json.loads(args.state_file.expanduser().resolve().read_text())
    for motion in args.motions:
        entry = state.get("motions", {}).get(motion, {})
        checkpoint_text = entry.get("final_checkpoint")
        run_text = entry.get("run_directory")
        if not checkpoint_text or not run_text:
            print(f"[SKIP] {motion}: no completed checkpoint in state")
            continue
        checkpoint = Path(checkpoint_text)
        run_dir = Path(run_text)
        evaluation_dir = args.output_root.expanduser().resolve() / motion / "evaluation"
        detailed_path = evaluation_dir / "tracking_errors_detailed.csv"
        if detailed_path.is_file() and not args.force:
            print(f"[SKIP] {motion}: detailed evaluation already exists")
            continue
        command = [
            sys.executable,
            "scripts/rsl_rl/play.py",
            "--task",
            "Tracking-Flat-G1-v0",
            "--motion_file",
            str(MOTIONS[motion]),
            "--load_run",
            run_dir.name,
            "--checkpoint",
            checkpoint.name,
            "--evaluation_output",
            str(evaluation_dir),
            "--headless",
        ]
        if args.video:
            command.append("--video")
        print("[RUN] " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
