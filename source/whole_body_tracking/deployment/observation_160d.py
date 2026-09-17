"""Pure NumPy observation contract for the 160D G1 robot-reference policy.

This module has no ROS, DDS, ONNX, or robot-control side effects. It is the
single source of truth shared by offline tests and the real-robot adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


NUM_JOINTS = 29
OBS_DIM = 160
TORSO_BODY_INDEX = 9

JOINT_SEQ = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

JOINT_XML = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

OBS_SLICES: Mapping[str, slice] = {
    "command": slice(0, 58),
    "motion_anchor_pos_b": slice(58, 61),
    "motion_anchor_ori_b": slice(61, 67),
    "base_lin_vel_b": slice(67, 70),
    "base_ang_vel_b": slice(70, 73),
    "joint_pos_rel": slice(73, 102),
    "joint_vel": slice(102, 131),
    "last_action": slice(131, 160),
}


def _vector(name: str, value, size: int) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32).reshape(-1)
    if out.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return out


def normalize_quat_wxyz(value, *, name: str = "quaternion") -> np.ndarray:
    quat = _vector(name, value, 4).astype(np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError(f"{name} has near-zero norm")
    return (quat / norm).astype(np.float32)


def quat_wxyz_to_matrix(value) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(value).astype(np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def relative_anchor_position_b(current_pos_w, current_quat_wxyz, reference_pos_w) -> np.ndarray:
    """Return R_current.T @ (reference_position - current_position)."""
    current_pos = _vector("current_anchor_pos_w", current_pos_w, 3).astype(np.float64)
    reference_pos = _vector("reference_anchor_pos_w", reference_pos_w, 3).astype(np.float64)
    rotation_w = quat_wxyz_to_matrix(current_quat_wxyz)
    return (rotation_w.T @ (reference_pos - current_pos)).astype(np.float32)


def relative_anchor_orientation_6d(current_quat_wxyz, reference_quat_wxyz) -> np.ndarray:
    """Return first two columns of R_current.T @ R_reference, row-major flattened."""
    current = quat_wxyz_to_matrix(current_quat_wxyz)
    reference = quat_wxyz_to_matrix(reference_quat_wxyz)
    relative = current.T @ reference
    return relative[:, :2].reshape(-1).astype(np.float32)


def reorder(values, source_names: Sequence[str], target_names: Sequence[str], *, name: str) -> np.ndarray:
    values = _vector(name, values, len(source_names))
    if len(set(source_names)) != len(source_names):
        raise ValueError(f"{name} source joint names contain duplicates")
    missing = [joint for joint in target_names if joint not in source_names]
    if missing:
        raise ValueError(f"{name} is missing joints: {missing}")
    return np.asarray([values[source_names.index(joint)] for joint in target_names], dtype=np.float32)


@dataclass(frozen=True)
class ObservationInputs:
    reference_joint_pos_seq: np.ndarray
    reference_joint_vel_seq: np.ndarray
    current_anchor_pos_w: np.ndarray
    current_anchor_position_frame_quat_wxyz: np.ndarray
    current_anchor_quat_wxyz: np.ndarray
    reference_anchor_pos_w: np.ndarray
    reference_anchor_quat_wxyz: np.ndarray
    base_lin_vel_b: np.ndarray
    base_ang_vel_b: np.ndarray
    current_joint_pos_seq: np.ndarray
    current_joint_vel_seq: np.ndarray
    default_joint_pos_seq: np.ndarray
    last_action: np.ndarray


def build_observation_160d(fields: ObservationInputs) -> np.ndarray:
    """Pack the exact policy observation order used by Tracking-Flat-G1-v0."""
    reference_q = _vector("reference_joint_pos_seq", fields.reference_joint_pos_seq, NUM_JOINTS)
    reference_dq = _vector("reference_joint_vel_seq", fields.reference_joint_vel_seq, NUM_JOINTS)
    current_q = _vector("current_joint_pos_seq", fields.current_joint_pos_seq, NUM_JOINTS)
    current_dq = _vector("current_joint_vel_seq", fields.current_joint_vel_seq, NUM_JOINTS)
    default_q = _vector("default_joint_pos_seq", fields.default_joint_pos_seq, NUM_JOINTS)

    parts = (
        np.concatenate((reference_q, reference_dq)),
        relative_anchor_position_b(
            fields.current_anchor_pos_w,
            fields.current_anchor_position_frame_quat_wxyz,
            fields.reference_anchor_pos_w,
        ),
        relative_anchor_orientation_6d(
            fields.current_anchor_quat_wxyz,
            fields.reference_anchor_quat_wxyz,
        ),
        _vector("base_lin_vel_b", fields.base_lin_vel_b, 3),
        _vector("base_ang_vel_b", fields.base_ang_vel_b, 3),
        current_q - default_q,
        current_dq,
        _vector("last_action", fields.last_action, NUM_JOINTS),
    )
    observation = np.concatenate(parts).astype(np.float32, copy=False)
    if observation.shape != (OBS_DIM,):
        raise RuntimeError(f"packed observation has shape {observation.shape}, expected ({OBS_DIM},)")
    if not np.isfinite(observation).all():
        raise RuntimeError("packed observation contains NaN or Inf")
    return observation


def validate_motion_npz(motion) -> int:
    required = (
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w", "fps",
    )
    missing = [key for key in required if key not in motion.files]
    if missing:
        raise ValueError(f"motion NPZ missing fields: {missing}")
    frames = int(len(motion["joint_pos"]))
    if frames < 2:
        raise ValueError(f"motion must contain at least 2 frames, got {frames}")
    for key in required[:-1]:
        array = np.asarray(motion[key])
        if len(array) != frames:
            raise ValueError(f"motion field {key} length {len(array)} != {frames}")
        if not np.isfinite(array).all():
            raise ValueError(f"motion field {key} contains NaN or Inf")
    if np.asarray(motion["joint_pos"]).shape[1] != NUM_JOINTS:
        raise ValueError("motion joint_pos must contain 29 joints")
    if np.asarray(motion["joint_vel"]).shape[1] != NUM_JOINTS:
        raise ValueError("motion joint_vel must contain 29 joints")
    if np.asarray(motion["body_pos_w"]).shape[1] <= TORSO_BODY_INDEX:
        raise ValueError("motion does not contain torso body index 9")
    return frames
