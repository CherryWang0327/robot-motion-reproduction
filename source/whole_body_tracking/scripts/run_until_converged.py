#!/usr/bin/env python3
"""Train one WBT motion in restartable segments until convergence is detected."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from analyze_training_convergence import analyze


ROOT = Path(__file__).resolve().parents[1]


def checkpoints(run_dir: Path) -> list[Path]:
    valid = []
    for path in run_dir.glob("model_*.pt"):
        if not path.stem.removeprefix("model_").isdigit():
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if all(not torch.is_tensor(value) or torch.isfinite(value).all().item()
                   for value in checkpoint.get("model_state_dict", {}).values()):
                valid.append(path)
        except (OSError, RuntimeError, ValueError, KeyError):
            pass
    return sorted(valid, key=lambda path: int(path.stem.removeprefix("model_")))


def iteration(path: Path) -> int:
    return int(path.stem.removeprefix("model_"))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True, help="Input G1 reference NPZ")
    parser.add_argument("--run-dir", type=Path, required=True, help="Output/resume checkpoint directory")
    parser.add_argument("--run-name", required=True, help="TensorBoard run name")
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segment-iterations", type=int, default=10_000)
    parser.add_argument("--min-iterations", type=int, default=30_000)
    parser.add_argument("--window", type=int, default=1_000)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()

    motion_file = args.motion_file.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if not motion_file.is_file():
        parser.error(f"--motion-file does not exist: {motion_file}")
    if min(args.segment_iterations, args.min_iterations, args.window) <= 0:
        parser.error("segment/window/min iterations must be positive")

    state_path = run_dir / "until_converged_state.json"
    state = {
        "stage": "training", "motion_file": str(motion_file), "run_directory": str(run_dir),
        "min_iterations": args.min_iterations, "window": args.window,
        "segment_iterations": args.segment_iterations,
        "rule": "continue in segments until convergence",
    }
    while True:
        completed = checkpoints(run_dir) if run_dir.is_dir() else []
        latest = completed[-1] if completed else None
        start_iteration = iteration(latest) if latest else 0
        target = start_iteration + args.segment_iterations
        # A repeated command against an already converged run must be a no-op,
        # not a new training segment appended to its final checkpoint.
        if latest is not None:
            try:
                existing_report = analyze(run_dir, target, args.window, args.min_iterations)
                if existing_report["converged"]:
                    state.update(stage="complete", latest_iteration=existing_report["iteration"],
                                 latest_checkpoint=existing_report["latest_checkpoint"],
                                 final_checkpoint=str(latest), final_iteration=start_iteration,
                                 converged=True, training_pid=None)
                    save_state(state_path, state)
                    print(f"[QUEUE] Already converged at {latest.name}; nothing to start.", flush=True)
                    return
            except (OSError, ValueError):
                pass
        command = [sys.executable, "scripts/rsl_rl/train.py", "--task", args.task,
                   "--motion_file", str(motion_file), "--num_envs", str(args.num_envs),
                   "--max_iterations", str(target), "--seed", str(args.seed), "--headless",
                   "--logger", "tensorboard", "--run_name", args.run_name,
                   "--resume_log_dir", str(run_dir)]
        if latest:
            command.extend(["--resume_checkpoint", str(latest)])
        print(f"[QUEUE] Starting {args.run_name}: {start_iteration} -> {target}", flush=True)
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        state.update(training_pid=process.pid, latest_checkpoint=str(latest) if latest else None,
                     segment_target=target)
        save_state(state_path, state)
        checkpoint_target = None
        while process.poll() is None:
            try:
                report = analyze(run_dir, target, args.window, args.min_iterations)
                state.update(latest_iteration=report["iteration"], latest_checkpoint=report["latest_checkpoint"],
                             plateau=report["plateau"], converged=report["converged"])
                save_state(state_path, state)
                if report["converged"] and checkpoint_target is None:
                    checkpoint_target = int(math.ceil(report["iteration"] / 500.0) * 500)
                    print(f"[QUEUE] Converged; waiting for model_{checkpoint_target}.pt", flush=True)
                if checkpoint_target is not None and checkpoints(run_dir) and iteration(checkpoints(run_dir)[-1]) >= checkpoint_target:
                    stop(process)
                    state.update(stage="complete", final_checkpoint=str(checkpoints(run_dir)[-1]),
                                 final_iteration=iteration(checkpoints(run_dir)[-1]), training_pid=None)
                    save_state(state_path, state)
                    return
            except (OSError, ValueError) as exc:
                print(f"[QUEUE] Waiting for metrics: {exc}", flush=True)
            time.sleep(max(args.poll_seconds, 1))
        state["training_pid"] = None
        save_state(state_path, state)
        print("[QUEUE] Segment ended without convergence; resuming from latest checkpoint.", flush=True)


if __name__ == "__main__":
    main()
