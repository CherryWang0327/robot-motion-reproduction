#!/usr/bin/env python3
"""Watch overall and current-run status for the ten-motion baseline orchestrator."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from analyze_training_convergence import analyze
from run_ten_motion_baseline import MOTIONS, RESULT_ROOT


def _print_once(state_file: Path) -> None:
    if not state_file.is_file():
        raise ValueError(f"state file does not exist; start the orchestrator first: {state_file}")
    state = json.loads(state_file.read_text())
    protocol = state.get("protocol", {})
    print(f"Updated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(
        "Protocol: "
        f"envs={protocol.get('num_envs')} seed={protocol.get('seed')} "
        f"min={protocol.get('min_iterations')} max={protocol.get('max_iterations')}"
    )
    print()
    print(f"{'motion':30} {'stage':12} {'iteration':>10} {'converged':>10} checkpoint")
    print("-" * 100)
    current_details = None
    complete_count = 0
    for motion in MOTIONS:
        entry = state.get("motions", {}).get(motion, {})
        stage = entry.get("stage", "pending")
        if stage == "complete":
            complete_count += 1
        iteration = entry.get("latest_iteration", entry.get("final_iteration", "-"))
        converged = entry.get("converged", "-")
        checkpoint = Path(entry["final_checkpoint"]).name if entry.get("final_checkpoint") else "-"
        print(f"{motion:30} {stage:12} {str(iteration):>10} {str(converged):>10} {checkpoint}")
        if stage == "training" and entry.get("run_directory"):
            current_details = (motion, Path(entry["run_directory"]))
    print("-" * 100)
    print(f"Completed: {complete_count}/{len(MOTIONS)}")

    if current_details:
        motion, run_dir = current_details
        try:
            report = analyze(
                run_dir,
                int(protocol.get("max_iterations", 100000)),
                int(protocol.get("window", 1000)),
                int(protocol.get("min_iterations", 30000)),
            )
        except (OSError, ValueError) as exc:
            print(f"\nCurrent motion {motion}: waiting for complete metrics ({exc})")
            return
        eta = dt.timedelta(seconds=round(report["eta_seconds"]))
        print(f"\nCurrent motion: {motion}")
        print(
            f"iteration={report['iteration']}/{report['max_iterations']} "
            f"progress={report['progress_percent']:.1f}% fps={report['steps_per_second']:.0f}"
        )
        print(f"ETA to maximum budget={eta}; estimated finish={report['estimated_finish']}")
        print(
            f"plateau={report['plateau']} converged={report['converged']} "
            f"latest_checkpoint={report['latest_checkpoint']}"
        )
        for reason in report["reasons"]:
            print(f"not converged: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state_file", type=Path, default=RESULT_ROOT / "orchestrator_state.json")
    parser.add_argument("--watch", type=float, help="Refresh interval in seconds; omit for one snapshot")
    args = parser.parse_args()
    while True:
        try:
            _print_once(args.state_file.expanduser().resolve())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"status error: {exc}")
        if args.watch is None:
            break
        print("\n" + "=" * 100 + "\n", flush=True)
        time.sleep(max(args.watch, 1.0))


if __name__ == "__main__":
    main()
