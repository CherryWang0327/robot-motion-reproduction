#!/usr/bin/env python3
"""Find and group peak tracking-error times for completed baseline evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_body_pos",
    "error_body_rot",
    "error_joint_pos",
    "error_joint_vel",
)


def analyze_motion(motion_dir: Path, fps: float, boundary_seconds: float, cluster_seconds: float) -> dict:
    error_path = motion_dir / "evaluation" / "tracking_errors.csv"
    video_path = motion_dir / "evaluation" / "video" / "rl-video-step-0.mp4"
    with error_path.open() as file:
        rows = list(csv.DictReader(file))
    duration = len(rows) / fps
    peaks = []
    for metric in METRICS:
        row = max(rows, key=lambda item: float(item[metric]))
        step = int(row["step"])
        time_seconds = step / fps
        boundary = None
        if time_seconds < boundary_seconds:
            boundary = "initialization"
        elif duration - time_seconds <= boundary_seconds:
            boundary = "motion_end"
        peaks.append(
            {
                "metric": metric,
                "step": step,
                "time_seconds": time_seconds,
                "maximum": float(row[metric]),
                "boundary": boundary,
            }
        )

    clusters = []
    for peak in sorted(peaks, key=lambda item: item["time_seconds"]):
        if not clusters or peak["time_seconds"] - clusters[-1]["last_time"] > cluster_seconds:
            clusters.append({"first_time": peak["time_seconds"], "last_time": peak["time_seconds"], "peaks": [peak]})
        else:
            clusters[-1]["last_time"] = peak["time_seconds"]
            clusters[-1]["peaks"].append(peak)
    for cluster in clusters:
        center = sum(peak["time_seconds"] for peak in cluster["peaks"]) / len(cluster["peaks"])
        start = max(0.0, center - 1.0)
        clip_duration = min(2.5, duration - start)
        cluster.update(
            center_time=center,
            start_time=start,
            clip_duration=clip_duration,
            command=(
                f'ffplay -autoexit -ss {start:.2f} -t {clip_duration:.2f} '
                f'-vf "setpts=4.0*PTS" "{video_path.resolve()}"'
            ),
        )
    return {
        "motion": motion_dir.name,
        "frames": len(rows),
        "fps": fps,
        "duration_seconds": duration,
        "tracking_errors": str(error_path.resolve()),
        "video": str(video_path.resolve()),
        "peaks": peaks,
        "clusters": clusters,
    }


def write_markdown(reports: list[dict], output_path: Path) -> None:
    lines = ["# Evaluation Peak Inspection Report", ""]
    for report in reports:
        lines.extend(
            [
                f"## {report['motion']}",
                "",
                f"Duration: {report['duration_seconds']:.2f} s",
                "",
                "| Metric | Peak time | Maximum | Boundary note |",
                "|---|---:|---:|---|",
            ]
        )
        for peak in report["peaks"]:
            lines.append(
                f"| {peak['metric']} | {peak['time_seconds']:.2f} s | {peak['maximum']:.6f} | "
                f"{peak['boundary'] or ''} |"
            )
        lines.extend(["", "Grouped slow-motion inspections:", ""])
        for index, cluster in enumerate(report["clusters"], 1):
            metrics = ", ".join(peak["metric"] for peak in cluster["peaks"])
            boundary_notes = sorted({peak["boundary"] for peak in cluster["peaks"] if peak["boundary"]})
            note = f" ({', '.join(boundary_notes)})" if boundary_notes else ""
            lines.extend(
                [
                    f"{index}. `{metrics}` around {cluster['center_time']:.2f} s{note}",
                    "",
                    "```bash",
                    cluster["command"],
                    "```",
                    "",
                ]
            )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, default=Path("results/baseline"))
    parser.add_argument("--motions", nargs="*", help="Defaults to every completed evaluation")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--boundary_seconds", type=float, default=0.5)
    parser.add_argument("--cluster_seconds", type=float, default=0.75)
    parser.add_argument("--output_dir", type=Path, default=Path("results/baseline/diagnostics"))
    args = parser.parse_args()

    if args.motions:
        motion_dirs = [args.result_root / motion for motion in args.motions]
    else:
        motion_dirs = sorted(path.parent.parent for path in args.result_root.glob("*/evaluation/tracking_errors.csv"))
    missing = [str(path) for path in motion_dirs if not (path / "evaluation" / "tracking_errors.csv").is_file()]
    if missing:
        parser.error("missing completed evaluations: " + ", ".join(missing))

    reports = [analyze_motion(path, args.fps, args.boundary_seconds, args.cluster_seconds) for path in motion_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "evaluation_peaks.json"
    markdown_path = args.output_dir / "evaluation_peaks.md"
    json_path.write_text(json.dumps(reports, indent=2) + "\n")
    write_markdown(reports, markdown_path)
    print(f"JSON report: {json_path.resolve()}")
    print(f"Inspection commands: {markdown_path.resolve()}")


if __name__ == "__main__":
    main()
