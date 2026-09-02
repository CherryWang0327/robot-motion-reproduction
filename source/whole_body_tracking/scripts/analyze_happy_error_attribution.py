#!/usr/bin/env python3
"""CPU-only error attribution for Happy GMR vs ProtoMotions/PyRoki."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "retarget_comparison" / "happy_error_attribution"
GMR_NPZ = ROOT / "inputs" / "happy" / "happy_gmr_50fps.npz"
PROTO_NPZ = (
    ROOT
    / "retarget_comparison"
    / "protomotions"
    / "happy"
    / "04_wbt_npz"
    / "happy_smpl_50fps_pyroki_proto.npz"
)
GMR_SUMMARY = ROOT / "results" / "baseline" / "happy" / "evaluation" / "summary.json"
PROTO_SUMMARY = (
    ROOT / "results" / "retarget_comparison" / "proto_happy" / "evaluation" / "summary.json"
)
PROTO_DETAILED = PROTO_SUMMARY.parent / "tracking_errors_detailed.csv"

# Isaac Lab articulation order used by this asset: root, then the interleaved
# joint order visible in deterministic evaluation (left/right/waist, etc.).
BODY_INDEX = {
    "pelvis": 0,
    "left_ankle_roll_link": 18,
    "right_ankle_roll_link": 19,
    "torso_link": 9,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def scalar_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "bias_corrected_rmse": float(np.sqrt(np.mean((values - values.mean()) ** 2))),
    }


def vector_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    magnitudes = np.linalg.norm(values, axis=-1)
    return {
        "component_rms": float(np.sqrt(np.mean(values**2))),
        "mean_magnitude": float(magnitudes.mean()),
        "magnitude_rms": float(np.sqrt(np.mean(magnitudes**2))),
        "peak_component_absolute": float(np.max(np.abs(values))),
        "peak_magnitude": float(magnitudes.max()),
    }


def reference_metrics(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    fps = float(arrays["fps"].reshape(-1)[0])
    dt = 1.0 / fps
    joint_vel = arrays["joint_vel"]
    joint_acc = np.gradient(joint_vel, dt, axis=0)
    joint_jerk = np.gradient(joint_acc, dt, axis=0)
    body_pos = arrays["body_pos_w"]
    result = {
        "fps": fps,
        "frames": int(joint_vel.shape[0]),
        "duration_seconds": float(joint_vel.shape[0] / fps),
        "joint_velocity": vector_stats(joint_vel),
        "joint_acceleration": vector_stats(joint_acc),
        "joint_jerk": vector_stats(joint_jerk),
        "pelvis_linear_velocity": vector_stats(arrays["body_lin_vel_w"][:, BODY_INDEX["pelvis"]]),
        "pelvis_angular_velocity": vector_stats(arrays["body_ang_vel_w"][:, BODY_INDEX["pelvis"]]),
        "torso_linear_velocity": vector_stats(arrays["body_lin_vel_w"][:, BODY_INDEX["torso_link"]]),
        "torso_angular_velocity": vector_stats(arrays["body_ang_vel_w"][:, BODY_INDEX["torso_link"]]),
        "height": {},
    }
    for name, index in BODY_INDEX.items():
        z = body_pos[:, index, 2]
        result["height"][name] = {
            "initial": float(z[0]),
            "mean": float(z.mean()),
            "std": float(z.std()),
            "min": float(z.min()),
            "max": float(z.max()),
            "range": float(np.ptp(z)),
        }
    return result, arrays


def load_proto_anchor_errors() -> dict[str, dict[str, float]]:
    with PROTO_DETAILED.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for axis in "xyz":
        # play.py stores reference - policy; requested bias convention is policy - reference.
        policy_minus_reference = -np.array(
            [float(row[f"anchor_pos_error_{axis}"]) for row in rows], dtype=np.float64
        )
        result[axis] = scalar_stats(policy_minus_reference)
    return result


def plot_trajectories(gmr: dict[str, np.ndarray], proto: dict[str, np.ndarray], fps: float) -> None:
    n = min(len(gmr["joint_pos"]), len(proto["joint_pos"]))
    time = np.arange(n) / fps
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for name, index in BODY_INDEX.items():
        if name in {"pelvis", "torso_link"}:
            axes[0].plot(time, gmr["body_pos_w"][:n, index, 2], label=f"GMR {name}")
            axes[0].plot(time, proto["body_pos_w"][:n, index, 2], "--", label=f"PyRoki {name}")
    axes[0].set_ylabel("reference Z (m)")
    axes[0].legend(ncol=2, fontsize=8)
    for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        index = BODY_INDEX[name]
        axes[1].plot(time, gmr["body_pos_w"][:n, index, 2], label=f"GMR {name}")
        axes[1].plot(time, proto["body_pos_w"][:n, index, 2], "--", label=f"PyRoki {name}")
    axes[1].set_ylabel("ankle Z (m)")
    axes[1].legend(ncol=2, fontsize=8)
    axes[2].plot(time, np.linalg.norm(gmr["joint_vel"][:n], axis=1), label="GMR")
    axes[2].plot(time, np.linalg.norm(proto["joint_vel"][:n], axis=1), label="PyRoki")
    axes[2].set_ylabel("joint velocity L2 (rad/s)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "reference_trajectories.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gmr_summary = json.loads(GMR_SUMMARY.read_text())
    proto_summary = json.loads(PROTO_SUMMARY.read_text())
    gmr_reference, gmr_arrays = reference_metrics(GMR_NPZ)
    proto_reference, proto_arrays = reference_metrics(PROTO_NPZ)
    reference_height_difference = {}
    for name, index in BODY_INDEX.items():
        difference = proto_arrays["body_pos_w"][:, index, 2] - gmr_arrays["body_pos_w"][:, index, 2]
        reference_height_difference[name] = scalar_stats(difference)
    anchor_decomposition = {
        "sign_convention": "policy_minus_reference",
        "gmr": {
            "status": "not_available",
            "reason": "legacy GMR evaluation stored only vector norms, not signed XYZ trajectory",
        },
        "pyroki": load_proto_anchor_errors(),
        "body_position_signed_xyz": {
            "status": "not_available",
            "reason": "evaluation stored per-body position norms, not signed XYZ components",
        },
    }
    plot_trajectories(gmr_arrays, proto_arrays, gmr_reference["fps"])
    manifest_paths = [
        GMR_NPZ,
        PROTO_NPZ,
        GMR_SUMMARY,
        PROTO_SUMMARY,
        PROTO_DETAILED,
        ROOT / "retarget_comparison/protomotions/happy/source/happy_smpl_50fps.csv",
        ROOT / "scripts/run_protomotions_comparison_pipeline.py",
        ROOT / "scripts/run_retarget_comparison_training.py",
        ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py",
        Path("/home/unitree/projects/ProtoMotions/data/scripts/all_convert_amass_to_proto.py"),
        Path("/home/unitree/projects/ProtoMotions/pyroki/batch_retarget_to_g1_from_keypoints.py"),
        Path("/home/unitree/projects/ProtoMotions/data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py"),
    ]
    result = {
        "analysis": "Happy GMR vs ProtoMotions/PyRoki error attribution",
        "cpu_only": True,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in manifest_paths],
        "checkpoints": {
            "gmr": gmr_summary["checkpoint"],
            "pyroki": proto_summary["checkpoint"],
        },
        "commits": {
            "whole_body_tracking": git_head(ROOT),
            "projects_protomotions": git_head(Path("/home/unitree/projects/ProtoMotions")),
        },
        "evaluation_rmse": {
            "gmr": {key: value["rmse"] for key, value in gmr_summary["metrics"].items()},
            "pyroki": {key: value["rmse"] for key, value in proto_summary["metrics"].items()},
        },
        "completion": {
            "gmr": {
                "recorded": gmr_summary["recorded_steps"],
                "planned": gmr_summary["planned_steps"],
                "without_termination": gmr_summary["completed_without_termination"],
            },
            "pyroki": {
                "recorded": proto_summary["recorded_steps"],
                "planned": proto_summary["planned_steps"],
                "without_termination": proto_summary["completed_without_termination"],
            },
        },
        "anchor_signed_error_decomposition": anchor_decomposition,
        "reference_only_dynamics": {
            "gmr": gmr_reference,
            "pyroki": proto_reference,
            "pyroki_minus_gmr_height": reference_height_difference,
        },
        "body_index_provenance": {
            "mapping": BODY_INDEX,
            "basis": "Isaac articulation order corroborated by evaluation joint order and physical height sanity checks",
        },
        "contact_metrics": {
            "status": "not_available_for_fair_reference_only_comparison",
            "reason": "WBT NPZ contains kinematics but no contact labels; height alone is not a reliable contact label",
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(OUT / "metrics.json")


if __name__ == "__main__":
    main()
