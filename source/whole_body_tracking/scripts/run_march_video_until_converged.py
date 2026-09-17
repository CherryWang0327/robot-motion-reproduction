#!/usr/bin/env python3
"""Run GMR then PyRoki until each satisfies the TensorBoard convergence test.

Training is deliberately split into bounded, restartable segments.  Reaching a
segment limit is never treated as completion: the next segment resumes from the
latest complete checkpoint.  Only the convergence monitor advances the queue.
"""

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
STATE_PATH = ROOT / "results" / "march_video" / "until_converged_state.json"
POLL_SECONDS = 60
SEGMENT_ITERATIONS = 10_000
MIN_ITERATIONS = 30_000
WINDOW = 1_000
TASK = "Tracking-Flat-G1-v0"
MOTIONS = (
    (
        "march_video_gmr",
        ROOT / "inputs" / "march_video" / "march_video_gmr_50fps.npz",
        ROOT / "logs" / "rsl_rl" / "g1_flat" / "2026-09-09_04-16-52_baseline_march_video_gmr_100k",
    ),
    (
        "march_video_pyroki",
        ROOT / "inputs" / "march_video" / "proto_50fps" / "march_video_motionlib_pyroki_proto.npz",
        ROOT / "logs" / "rsl_rl" / "g1_flat" / "2026-09-04_10-22-28_march_video_pyroki",
    ),
)


def checkpoint_is_finite(path: Path) -> bool:
    """Reject checkpoints whose policy tensors were corrupted by a NaN update."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return all(
            not torch.is_tensor(value) or torch.isfinite(value).all().item()
            for value in checkpoint.get("model_state_dict", {}).values()
        )
    except (OSError, RuntimeError, ValueError, KeyError):
        return False


def checkpoints(run_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in run_dir.glob("model_*.pt")
            if path.stem.removeprefix("model_").isdigit() and checkpoint_is_finite(path)
        ),
        key=lambda path: int(path.stem.removeprefix("model_")),
    )


def iteration(checkpoint: Path) -> int:
    return int(checkpoint.stem.removeprefix("model_"))


def save(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE_PATH)


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
    parser.parse_args()
    protocol = {
            "min_iterations": MIN_ITERATIONS,
            "window": WINDOW,
            "segment_iterations": SEGMENT_ITERATIONS,
            "rule": "continue in 10000-iteration segments until convergence",
    }
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text())
        state["protocol"] = protocol
        state.setdefault("motions", {})
    else:
        state = {"protocol": protocol, "motions": {}}
    save(state)

    for name, motion_file, run_dir in MOTIONS:
        if not motion_file.is_file():
            raise FileNotFoundError(motion_file)
        if not run_dir.is_dir() or not checkpoints(run_dir):
            raise RuntimeError(f"{name} has no resumable run/checkpoint: {run_dir}")
        entry = state["motions"].setdefault(name, {"stage": "training", "run_directory": str(run_dir)})
        if entry.get("stage") == "complete":
            print(f"[QUEUE] {name} already converged; skipping.", flush=True)
            continue
        entry.update(stage="training", run_directory=str(run_dir))
        while True:
            latest = checkpoints(run_dir)[-1]
            # ``train.py`` needs a bounded target for one invocation, but the
            # scheduler has no global iteration cap: every unfinished segment
            # resumes from its newest checkpoint and adds another 10,000 steps.
            target = iteration(latest) + SEGMENT_ITERATIONS
            command = [
                sys.executable,
                "scripts/rsl_rl/train.py",
                "--task", TASK,
                "--motion_file", str(motion_file),
                "--num_envs", "4096",
                "--max_iterations", str(target),
                "--seed", "0",
                "--headless",
                "--logger", "tensorboard",
                "--run_name", name,
                "--resume_checkpoint", str(latest.resolve()),
                "--resume_log_dir", str(run_dir.resolve()),
            ]
            print(f"[QUEUE] Starting {name}: {iteration(latest)} -> {target}", flush=True)
            process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
            entry.update(training_pid=process.pid, latest_checkpoint=str(latest), segment_target=target)
            save(state)
            checkpoint_target = None
            converged_report = None
            while process.poll() is None:
                try:
                    report = analyze(run_dir, target, WINDOW, MIN_ITERATIONS)
                    entry.update(
                        latest_iteration=report["iteration"],
                        latest_checkpoint=report["latest_checkpoint"],
                        plateau=report["plateau"],
                        converged=report["converged"],
                    )
                    save(state)
                    if report["converged"] and checkpoint_target is None:
                        converged_report = report
                        checkpoint_target = int(math.ceil(report["iteration"] / 500.0) * 500)
                        print(f"[QUEUE] {name} converged; waiting for model_{checkpoint_target}.pt", flush=True)
                    if checkpoint_target is not None and checkpoints(run_dir):
                        if iteration(checkpoints(run_dir)[-1]) >= checkpoint_target:
                            stop(process)
                            break
                except (OSError, ValueError) as exc:
                    print(f"[QUEUE] {name}: waiting for metrics ({exc})", flush=True)
                time.sleep(POLL_SECONDS)

            if converged_report is not None:
                final_checkpoint = checkpoints(run_dir)[-1]
                entry.update(
                    stage="complete",
                    converged=True,
                    final_checkpoint=str(final_checkpoint),
                    final_iteration=iteration(final_checkpoint),
                    training_pid=None,
                )
                save(state)
                print(f"[QUEUE] {name} complete at {final_checkpoint.name}", flush=True)
                break

            # Normal segment completion, interruption, or a recoverable child
            # failure: never advance the queue.  Resume from the latest saved
            # checkpoint after a brief pause instead.
            latest_after = checkpoints(run_dir)[-1]
            entry.update(
                stage="training",
                latest_checkpoint=str(latest_after),
                latest_iteration=iteration(latest_after),
                training_pid=None,
            )
            save(state)
            print(f"[QUEUE] {name} not converged; resuming from {latest_after.name}", flush=True)
            time.sleep(10)

    state["stage"] = "complete"
    save(state)


if __name__ == "__main__":
    main()
