#!/usr/bin/env python3
"""Export reproducible reward/error curves and a convergence report from TensorBoard."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from analyze_training_convergence import ERROR_TAGS, analyze


def _load_scalars(log_dir: Path, tags: tuple[str, ...]) -> dict[str, list[tuple[int, float]]]:
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    missing = sorted(set(tags) - available)
    if missing:
        raise ValueError("missing TensorBoard scalars: " + ", ".join(missing))
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in tags
    }


def _write_csv(series: dict[str, list[tuple[int, float]]], output_path: Path) -> None:
    rows: dict[int, dict[str, float | int]] = {}
    for tag, values in series.items():
        for step, value in values:
            rows.setdefault(step, {"iteration": step})[tag] = value
    fieldnames = ["iteration", *series]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[step] for step in sorted(rows))


def _plot(series: dict[str, list[tuple[int, float]]], output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/whole_body_tracking_matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to export PNG curves") from exc

    reward_tag = "Train/mean_reward"
    steps, values = zip(*series[reward_tag])
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(steps, values, linewidth=1.2)
    axis.set(title="Training Reward", xlabel="Iteration", ylabel="Mean reward")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "reward_curve.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    for axis, tag in zip(axes.flat, ERROR_TAGS, strict=True):
        steps, values = zip(*series[tag])
        axis.plot(steps, values, linewidth=1.0)
        axis.set_title(tag.removeprefix("Metrics/motion/"))
        axis.set_ylabel("Error")
        axis.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("Iteration")
    axes[-1, 1].set_xlabel("Iteration")
    figure.suptitle("Training Tracking Errors")
    figure.tight_layout()
    figure.savefig(output_dir / "tracking_error_curve.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, help="Defaults to <log_dir>/training_report")
    parser.add_argument("--max_iterations", type=int, default=100000)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--min_iterations", type=int, default=30000)
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        parser.error(f"log directory does not exist: {args.log_dir}")
    output_dir = args.output_dir or args.log_dir / "training_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    tags = ("Train/mean_reward", "Train/mean_episode_length", *ERROR_TAGS)
    series = _load_scalars(args.log_dir, tags)
    report = analyze(args.log_dir, args.max_iterations, args.window, args.min_iterations)

    _write_csv(series, output_dir / "training_metrics.csv")
    _plot(series, output_dir)
    (output_dir / "convergence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Training report written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
