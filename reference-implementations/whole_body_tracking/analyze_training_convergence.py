#!/usr/bin/env python3
"""CPU-only TensorBoard health, ETA, and convergence monitor for tracking runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ERROR_TAGS = (
    "Metrics/motion/error_anchor_pos",
    "Metrics/motion/error_anchor_rot",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_body_rot",
    "Metrics/motion/error_joint_pos",
    "Metrics/motion/error_joint_vel",
)


def _series(accumulator: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    events = accumulator.Scalars(tag)
    if not events:
        raise ValueError(f"TensorBoard scalar is missing or empty: {tag}")
    return (
        np.asarray([event.step for event in events], dtype=np.int64),
        np.asarray([event.value for event in events], dtype=np.float64),
        np.asarray([event.wall_time for event in events], dtype=np.float64),
    )


def _window_mean(steps: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    selected = values[(steps >= start) & (steps < end)]
    if selected.size == 0:
        raise ValueError(f"no scalar samples in iteration window [{start}, {end})")
    return float(selected.mean())


def analyze(log_dir: Path, max_iterations: int, window: int, min_iterations: int) -> dict:
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    required = {
        "Perf/total_fps", "Train/mean_reward", "Train/mean_episode_length",
        "Loss/value_function", "Loss/surrogate", *ERROR_TAGS,
    }
    missing = sorted(required - available)
    if missing:
        raise ValueError("missing TensorBoard scalars: " + ", ".join(missing))

    perf_steps, perf_values, perf_times = _series(accumulator, "Perf/total_fps")
    iteration = int(perf_steps[-1])
    elapsed_iterations = max(iteration - int(perf_steps[0]), 1)
    seconds_per_iteration = float((perf_times[-1] - perf_times[0]) / elapsed_iterations)
    remaining_seconds = max(0, max_iterations - iteration - 1) * seconds_per_iteration
    finish_time = dt.datetime.fromtimestamp(perf_times[-1] + remaining_seconds).astimezone()

    recent_start = max(0, iteration - window + 1)
    previous_start = max(0, recent_start - window)
    # TensorBoard iterations are zero-based: step 29999 means 30000 iterations completed.
    enough_windows = recent_start - previous_start == window and iteration + 1 >= min_iterations

    metrics = {}
    all_finite = True
    for tag in required:
        steps, values, _ = _series(accumulator, tag)
        all_finite = all_finite and bool(np.isfinite(values).all())
        recent = _window_mean(steps, values, recent_start, iteration + 1)
        previous = None
        relative_change = None
        if enough_windows:
            previous = _window_mean(steps, values, previous_start, recent_start)
            relative_change = (recent - previous) / max(abs(previous), 1.0e-12)
        metrics[tag] = {
            "latest": float(values[-1]),
            "recent_mean": recent,
            "previous_mean": previous,
            "relative_change": relative_change,
        }

    reward_change = metrics["Train/mean_reward"]["relative_change"]
    error_changes = [metrics[tag]["relative_change"] for tag in ERROR_TAGS]
    episode_mean = metrics["Train/mean_episode_length"]["recent_mean"]
    plateau = bool(
        enough_windows
        and reward_change is not None
        and abs(reward_change) < 0.01
        and all(change is not None and abs(change) < 0.02 for change in error_changes)
    )
    converged = bool(plateau and episode_mean >= 495.0 and all_finite)
    reasons = []
    if not enough_windows:
        reasons.append(f"need completed iterations >= {max(min_iterations, 2 * window)} for two complete windows")
    elif not plateau:
        reasons.append("reward or tracking errors are still changing beyond plateau thresholds")
    if episode_mean < 495.0:
        reasons.append(f"recent mean episode length {episode_mean:.2f} is below 495")
    if not all_finite:
        reasons.append("one or more TensorBoard scalars contain NaN/Inf")

    checkpoints = sorted(
        log_dir.glob("model_*.pt"),
        key=lambda path: int(path.stem.removeprefix("model_")),
    )
    return {
        "log_dir": str(log_dir.resolve()),
        "iteration": iteration,
        "max_iterations": max_iterations,
        "progress_percent": 100.0 * iteration / max_iterations,
        "steps_per_second": float(perf_values[-1]),
        "seconds_per_iteration": seconds_per_iteration,
        "eta_seconds": remaining_seconds,
        "estimated_finish": finish_time.isoformat(timespec="seconds"),
        "latest_checkpoint": str(checkpoints[-1].resolve()) if checkpoints else None,
        "all_finite": all_finite,
        "plateau": plateau,
        "converged": converged,
        "reasons": reasons,
        "window_iterations": window,
        "metrics": metrics,
    }


def print_report(report: dict) -> None:
    eta = dt.timedelta(seconds=round(report["eta_seconds"]))
    print(
        f"iteration {report['iteration']}/{report['max_iterations']} "
        f"({report['progress_percent']:.1f}%), {report['steps_per_second']:.0f} steps/s"
    )
    print(f"ETA {eta}; estimated finish {report['estimated_finish']}")
    print(f"latest checkpoint: {report['latest_checkpoint']}")
    print(f"finite={report['all_finite']} plateau={report['plateau']} converged={report['converged']}")
    for tag in ("Train/mean_reward", "Train/mean_episode_length", *ERROR_TAGS):
        metric = report["metrics"][tag]
        change = metric["relative_change"]
        change_text = "n/a" if change is None else f"{100 * change:+.3f}%"
        print(f"  {tag}: latest={metric['latest']:.6g}, window_change={change_text}")
    for reason in report["reasons"]:
        print(f"  not converged: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--max_iterations", type=int, default=100000)
    parser.add_argument("--window", type=int, default=1000, help="Iterations per convergence comparison window")
    parser.add_argument("--min_iterations", type=int, default=30000)
    parser.add_argument("--watch", type=float, help="Refresh interval in seconds; omit for one report")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    if args.max_iterations <= 0 or args.window <= 0 or args.min_iterations < 0:
        parser.error("iteration arguments must be positive")
    if not args.log_dir.is_dir():
        parser.error(f"log directory does not exist: {args.log_dir}")
    while True:
        try:
            report = analyze(args.log_dir, args.max_iterations, args.window, args.min_iterations)
            print(json.dumps(report, indent=2) if args.json else "") if args.json else print_report(report)
        except (OSError, ValueError) as exc:
            print(f"monitor error: {exc}")
        if args.watch is None:
            break
        time.sleep(max(args.watch, 1.0))


if __name__ == "__main__":
    main()

