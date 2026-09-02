#!/usr/bin/env python3
"""Validate local motion NPZ files before launching expensive baseline training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def validate(path: Path) -> dict:
    result = {"path": str(path.resolve()), "valid": True, "problems": []}
    try:
        with np.load(path) as data:
            missing = [name for name in REQUIRED_FIELDS if name not in data]
            if missing:
                result["valid"] = False
                result["problems"].append("missing fields: " + ", ".join(missing))
                return result

            frames = int(data["joint_pos"].shape[0])
            frame_fields = REQUIRED_FIELDS[1:]
            mismatched = [name for name in frame_fields if data[name].shape[0] != frames]
            non_finite = [name for name in REQUIRED_FIELDS if not np.isfinite(data[name]).all()]
            if mismatched:
                result["valid"] = False
                result["problems"].append("frame count mismatch: " + ", ".join(mismatched))
            if non_finite:
                result["valid"] = False
                result["problems"].append("NaN/Inf in: " + ", ".join(non_finite))

            result.update(
                fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
                frames=frames,
                duration_seconds=frames / float(np.asarray(data["fps"]).reshape(-1)[0]),
                joint_count=int(data["joint_pos"].shape[1]),
                body_count=int(data["body_pos_w"].shape[1]),
                shapes={name: list(data[name].shape) for name in REQUIRED_FIELDS},
            )
    except (OSError, ValueError, IndexError) as exc:
        result["valid"] = False
        result["problems"].append(str(exc))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_files", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    reports = [validate(path) for path in args.motion_files]
    for report in reports:
        status = "PASS" if report["valid"] else "FAIL"
        details = "" if not report["valid"] else (
            f" | {report['frames']} frames | {report['duration_seconds']:.2f}s | "
            f"{report['joint_count']} joints | {report['body_count']} bodies"
        )
        print(f"{status}: {report['path']}{details}")
        for problem in report["problems"]:
            print(f"  - {problem}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
    if not all(report["valid"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
