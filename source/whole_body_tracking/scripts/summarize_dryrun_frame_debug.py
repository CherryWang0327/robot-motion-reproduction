#!/usr/bin/env python3
"""Summarize structured position/orientation/velocity lines in a dry-run log."""

import argparse
import json
import re

import numpy as np


VECTOR_FIELDS = {
    "motion_anchor_pos_b": "obs_motion_anchor_pos_b",
    "relative_orientation_6d": "obs_motion_anchor_ori_6d",
    "base_lin_vel_b": "obs_base_lin_vel_b",
    "base_ang_vel_b": "obs_base_ang_vel_b",
    "robot_odom_rpy_deg": "robot_odom_rpy_deg",
    "robot_torso_rpy_deg": "robot_torso_rpy_deg",
    "reference_torso_rpy_deg": "reference_torso_rpy_deg",
    "relative_torso_rpy_deg": "relative_torso_rpy_deg",
}
SCALAR_FIELDS = {
    "relative_rotation_deg": "relative_rotation_deg",
    "orientation_orthogonality_error": "ori_matrix_orth_error",
    "odom_vs_torso_angle_deg": "odom_vs_torso_angle_deg",
}


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--output")
    args = parser.parse_args()
    text = open(args.log, encoding="utf-8", errors="replace").read()
    report = {}
    for output_name, label in VECTOR_FIELDS.items():
        matches = re.findall(rf"^\s*{re.escape(label)}\s*=\s*\[([^]]+)\]", text, re.MULTILINE)
        report[output_name] = stats([np.fromstring(value, sep=",") for value in matches])
    for output_name, label in SCALAR_FIELDS.items():
        matches = re.findall(rf"^\s*{re.escape(label)}\s*=\s*([-+0-9.eE]+)", text, re.MULTILINE)
        report[output_name] = stats([float(value) for value in matches])
    late = [float(value) for value in re.findall(r"\[LOOP_LATE\].*?late_ms=([0-9.]+)", text)]
    report["loop_late_ms"] = stats(late) if late else {"count": 0}
    report["stale_odom_count"] = text.count("STALE ODOM") + text.count("missing or stale")
    report["exit_seen"] = bool(re.search(r"^Exit$", text, re.MULTILINE))
    output = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
