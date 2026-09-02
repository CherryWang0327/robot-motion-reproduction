#!/usr/bin/env python3
"""CPU-only numerical audit of GMR and ProtoMotions/PyRoki WBT references."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pipeline_audit"
GMR = {
    "1438": ROOT / "inputs/1438/1438_gmr_50fps.npz",
    "1439": ROOT / "inputs/1439/1439_gmr_50fps.npz",
    "happy": ROOT / "inputs/happy/happy_gmr_50fps.npz",
    "taichi1": ROOT / "inputs/taichi1/taichi1_gmr_50fps.npz",
    "zhu0129": ROOT / "inputs/zhu0129/zhu0129_gmr_50fps.npz",
    "zhu0201_02_whswap_v2": ROOT / "inputs/zhu0201_02_whswap_v2/zhu0201_02_whswap_v2_gmr_50fps.npz",
    "zhu0201_03": ROOT / "inputs/zhu0201_03/zhu0201_03_gmr_50fps.npz",
    "zhu0201_04": ROOT / "inputs/zhu0201_04/zhu0201_04_gmr_50fps.npz",
    "zhu0201_upright_50fps": ROOT / "inputs/zhu0201_upright_50fps/zhu0201_upright_50fps_gmr_50fps.npz",
    "zhu0202_01": ROOT / "inputs/zhu0202_01/zhu0202_01_gmr_50fps.npz",
}
PROTO = {
    name: ROOT / f"retarget_comparison/protomotions/{name}/04_wbt_npz/{name}_smpl_50fps_pyroki_proto.npz"
    for name in ("happy", "taichi1")
}
PROTO_DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint",
    "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ISAAC_DOF_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def quat_angle_step(q: np.ndarray, fps: float) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    dot = np.abs(np.sum(q[1:] * q[:-1], axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0)) * fps


def inspect_npz(path: Path) -> dict:
    with np.load(path) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    fps = float(arrays["fps"].reshape(-1)[0])
    joint_pos = arrays["joint_pos"].astype(np.float64)
    joint_vel = arrays["joint_vel"].astype(np.float64)
    body_pos = arrays["body_pos_w"].astype(np.float64)
    body_quat = arrays["body_quat_w"].astype(np.float64)
    body_lin = arrays["body_lin_vel_w"].astype(np.float64)
    body_ang = arrays["body_ang_vel_w"].astype(np.float64)
    dt = 1.0 / fps
    joint_fd = np.gradient(joint_pos, dt, axis=0)
    body_fd = np.gradient(body_pos, dt, axis=0)
    quat_norm_error = np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0)
    joint_vel_residual = joint_vel - joint_fd
    body_vel_residual = body_lin - body_fd
    foot_min_candidates = np.sort(body_pos.min(axis=0)[:, 2])[:4]
    return {
        "path": str(path.resolve()),
        "fps": fps,
        "frames": int(len(joint_pos)),
        "finite": bool(all(np.isfinite(v).all() for v in arrays.values())),
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "quaternion_norm_error_max": float(quat_norm_error.max()),
        "quaternion_norm_error_rms": rms(quat_norm_error),
        "root_quaternion_step_rate_rad_s_p99": float(np.quantile(quat_angle_step(body_quat[:, 0], fps), 0.99)),
        "root_quaternion_step_rate_rad_s_max": float(quat_angle_step(body_quat[:, 0], fps).max()),
        "joint_velocity_fd_residual_rms": rms(joint_vel_residual),
        "joint_velocity_fd_residual_p99_abs": float(np.quantile(np.abs(joint_vel_residual), 0.99)),
        "body_linear_velocity_fd_residual_rms": rms(body_vel_residual),
        "body_linear_velocity_fd_residual_p99_abs": float(np.quantile(np.abs(body_vel_residual), 0.99)),
        "root_z_initial": float(body_pos[0, 0, 2]),
        "root_z_mean": float(body_pos[:, 0, 2].mean()),
        "lowest_body_min_z_candidates": [float(v) for v in foot_min_candidates],
        "joint_position_abs_max": float(np.abs(joint_pos).max()),
        "joint_velocity_abs_max": float(np.abs(joint_vel).max()),
    }


def csv_frames(path: Path) -> int:
    with path.open(newline="") as stream:
        return sum(1 for _ in csv.reader(stream)) - 1


def inspect_pt_to_npz(name: str, npz_path: Path) -> dict:
    pt_path = ROOT / f"retarget_comparison/protomotions/{name}/03_g1_pyroki/{name}_smpl_50fps_pyroki.pt"
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    pt_joint = pt["dps"].detach().cpu().numpy()
    pt_root_pos = pt["gts"][:, 0].detach().cpu().numpy()
    pt_root_xyzw = pt["grs"][:, 0].detach().cpu().numpy()
    with np.load(npz_path) as data:
        npz_joint = np.asarray(data["joint_pos"])
        npz_root_pos = np.asarray(data["body_pos_w"][:, 0])
        npz_root_wxyz = np.asarray(data["body_quat_w"][:, 0])
    order = [PROTO_DOF_NAMES.index(joint) for joint in ISAAC_DOF_NAMES]
    frames = min(len(pt_joint), len(npz_joint))
    joint_delta = pt_joint[:frames, order] - npz_joint[:frames]
    pos_delta = pt_root_pos[:frames] - npz_root_pos[:frames]
    pt_root_wxyz = pt_root_xyzw[:frames, [3, 0, 1, 2]]
    dots = np.clip(np.abs(np.sum(pt_root_wxyz * npz_root_wxyz[:frames], axis=-1)), -1.0, 1.0)
    angle = 2.0 * np.arccos(dots)
    return {
        "pt_path": str(pt_path.resolve()),
        "pt_frames": int(len(pt_joint)),
        "npz_frames": int(len(npz_joint)),
        "joint_mapping": "Proto named DOF order reordered to Isaac articulation named DOF order",
        "joint_position_rms_rad": rms(joint_delta),
        "joint_position_max_abs_rad": float(np.abs(joint_delta).max()),
        "root_position_max_abs_m": float(np.abs(pos_delta).max()),
        "root_orientation_mean_rad": float(angle.mean()),
        "root_orientation_max_rad": float(angle.max()),
    }


def inspect_paired_coordinates(gmr_path: Path, pyroki_path: Path) -> dict:
    """Compare reference axes after removing the arbitrary initial translation."""
    with np.load(gmr_path) as data:
        gmr = np.asarray(data["body_pos_w"][:, 0], dtype=np.float64)
    with np.load(pyroki_path) as data:
        pyroki = np.asarray(data["body_pos_w"][:, 0], dtype=np.float64)
    frames = min(len(gmr), len(pyroki))
    gmr = gmr[:frames] - gmr[0]
    pyroki = pyroki[:frames] - pyroki[0]
    correlation = np.corrcoef(gmr.T, pyroki.T)[:3, 3:]
    # A +90 degree world-yaw maps PyRoki displacement into the GMR convention:
    # (x_gmr, y_gmr) ~= (y_pyroki, -x_pyroki).
    aligned_pyroki = np.column_stack((pyroki[:, 1], -pyroki[:, 0], pyroki[:, 2]))
    return {
        "frames": int(frames),
        "axis_order": ["x", "y", "z"],
        "correlation_rows_gmr_columns_pyroki": correlation.tolist(),
        "tested_fixed_yaw_relation": "gmr_xyz ~= (pyroki_y, -pyroki_x, pyroki_z)",
        "displacement_rms_before_alignment_m": rms(gmr - pyroki),
        "displacement_rms_after_fixed_yaw_alignment_m": rms(gmr - aligned_pyroki),
        "z_displacement_correlation": float(correlation[2, 2]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "gmr": {name: inspect_npz(path) for name, path in GMR.items()},
        "pyroki": {name: inspect_npz(path) for name, path in PROTO.items()},
        "paired_frame_alignment": {},
        "paired_coordinate_comparison": {},
        "pt_to_npz_name_based_mapping": {},
    }
    for name in PROTO:
        source_csv = ROOT / f"retarget_comparison/protomotions/{name}/source/{name}_smpl_50fps.csv"
        result["paired_frame_alignment"][name] = {
            "source_csv_frames": csv_frames(source_csv),
            "gmr_npz_frames": result["gmr"][name]["frames"],
            "pyroki_npz_frames": result["pyroki"][name]["frames"],
            "gmr_pyroki_equal": result["gmr"][name]["frames"] == result["pyroki"][name]["frames"],
        }
        result["pt_to_npz_name_based_mapping"][name] = inspect_pt_to_npz(name, PROTO[name])
        result["paired_coordinate_comparison"][name] = inspect_paired_coordinates(GMR[name], PROTO[name])
    (OUT / "motion_numerical_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(OUT / "motion_numerical_audit.json")


if __name__ == "__main__":
    main()
