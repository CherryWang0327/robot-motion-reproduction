#!/usr/bin/env python3
"""Create a readable cross-motion report from detailed evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def top_items(items: dict, section: str, count: int = 5) -> list[tuple[str, float]]:
    return sorted(
        ((name, values[section]["rmse"]) for name, values in items.items()), key=lambda item: item[1], reverse=True
    )[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, default=Path("results/baseline"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline/diagnostics/detailed_metrics.md"))
    args = parser.parse_args()

    summaries = sorted(args.result_root.glob("*/evaluation/summary.json"))
    reports = []
    for path in summaries:
        summary = json.loads(path.read_text())
        if "detailed_metrics" in summary:
            reports.append((path.parts[-3], summary["detailed_metrics"]))
    if not reports:
        parser.error("no detailed evaluation summaries found; run rerun_detailed_evaluations.py first")

    lines = ["# Detailed Evaluation Metrics", ""]
    for motion, details in reports:
        lines.extend([f"## {motion}", "", "### Anchor position by axis", ""])
        lines.extend(["| Axis | RMSE | P95 absolute | Max absolute |", "|---|---:|---:|---:|"])
        for axis, values in details["anchor_position_axes"].items():
            lines.append(
                f"| {axis} | {values['rmse']:.6f} | {values['p95_absolute']:.6f} | "
                f"{values['max_absolute']:.6f} |"
            )
        lines.extend(["", "### Largest joint errors", ""])
        lines.append("Position RMSE: " + ", ".join(f"`{name}`={value:.4f}" for name, value in top_items(details["joints"], "position_error")))
        lines.append("")
        lines.append("Velocity RMSE: " + ", ".join(f"`{name}`={value:.4f}" for name, value in top_items(details["joints"], "velocity_error")))
        lines.extend(["", "### Largest body errors", ""])
        lines.append("Position RMSE: " + ", ".join(f"`{name}`={value:.4f}" for name, value in top_items(details["bodies"], "position_error")))
        lines.append("")
        lines.append("Rotation RMSE: " + ", ".join(f"`{name}`={value:.4f}" for name, value in top_items(details["bodies"], "rotation_error")))
        lines.extend(["", "### Feet, energy, and limits", ""])
        lines.extend(["| Foot | Contact fraction | Contact slip proxy (m) | Max contact force |", "|---|---:|---:|---:|"])
        for foot, values in details["feet"].items():
            lines.append(
                f"| {foot} | {values['contact_fraction']:.4f} | {values['slip_distance_during_contact']:.6f} | "
                f"{values['contact_force_max']:.2f} |"
            )
        lines.extend(
            [
                "",
                f"Mean absolute joint power: {details['energy']['mean_absolute_joint_power']:.4f}",
                "",
                f"Absolute joint energy: {details['energy']['absolute_joint_energy']:.4f}",
                "",
                f"Minimum soft-limit margin: {details['joint_limits']['minimum_soft_limit_margin']:.6f}",
                "",
                f"Steps outside soft limits: {details['joint_limits']['steps_outside_soft_limits']}",
                "",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Detailed report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
