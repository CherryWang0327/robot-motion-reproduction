"""Canonical, dependency-light interchange for GVHMR global SMPL parameters."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PREFIX = "smpl_params_global_"
CSV_COLUMNS = (
    ["fps"]
    + [f"{PREFIX}global_orient_{i}" for i in range(3)]
    + [f"{PREFIX}body_pose_{i}" for i in range(63)]
    + [f"{PREFIX}transl_{i}" for i in range(3)]
    + [f"{PREFIX}betas_{i}" for i in range(10)]
)


def _numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    try:
        result = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
    except Exception as exc:
        raise ValueError(f"{name} cannot be converted to a NumPy array: {exc}") from exc
    if not np.issubdtype(result.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype {result.dtype}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def normalize_smpl_params(params: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate canonical SMPL fields and return frame-major float arrays."""
    required = ("global_orient", "body_pose", "transl", "betas")
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(f"smpl_params_global is missing required field(s): {', '.join(missing)}")

    orient = _numpy(params["global_orient"], "global_orient")
    pose = _numpy(params["body_pose"], "body_pose")
    transl = _numpy(params["transl"], "transl")
    betas = _numpy(params["betas"], "betas")
    if orient.ndim != 2 or orient.shape[1] != 3:
        raise ValueError(f"global_orient must have shape [T, 3], got {orient.shape}")
    if pose.ndim == 3 and pose.shape[1:] == (21, 3):
        pose = pose.reshape(pose.shape[0], 63)
    elif pose.ndim != 2 or pose.shape[1] != 63:
        raise ValueError(f"body_pose must have shape [T, 63] or [T, 21, 3], got {pose.shape}")
    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError(f"transl must have shape [T, 3], got {transl.shape}")

    frames = orient.shape[0]
    if frames == 0:
        raise ValueError("SMPL motion contains no frames")
    for name, value in (("body_pose", pose), ("transl", transl)):
        if value.shape[0] != frames:
            raise ValueError(
                f"frame count mismatch: global_orient has {frames} frames but {name} has {value.shape[0]}"
            )
    if betas.shape == (10,):
        betas = np.broadcast_to(betas, (frames, 10)).copy()
    elif betas.shape == (1, 10):
        betas = np.broadcast_to(betas, (frames, 10)).copy()
    elif betas.shape != (frames, 10):
        raise ValueError(f"betas must have shape [10], [1, 10], or [T, 10], got {betas.shape}")
    return {
        "global_orient": orient.astype(np.float32, copy=False),
        "body_pose": pose.astype(np.float32, copy=False),
        "transl": transl.astype(np.float32, copy=False),
        "betas": betas.astype(np.float32, copy=False),
    }


def _scalar_fps(value: Any, source: str) -> float:
    array = _numpy(value, source).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{source} must be a scalar FPS value, got shape {array.shape}")
    fps = float(array[0])
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    return fps


def resolve_fps(prediction: Mapping[str, Any], explicit_fps: float | None) -> float:
    found = []
    for key in ("fps", "video_fps", "mocap_frame_rate"):
        if key in prediction:
            found.append((key, _scalar_fps(prediction[key], key)))
    if found and any(not np.isclose(value, found[0][1]) for _, value in found[1:]):
        raise ValueError(f"inconsistent FPS metadata: {found}")
    if explicit_fps is not None:
        fps = _scalar_fps(explicit_fps, "--fps")
        if found and not np.isclose(fps, found[0][1]):
            raise ValueError(f"--fps {fps} conflicts with {found[0][0]}={found[0][1]} in the PT file")
        return fps
    if found:
        return found[0][1]
    raise ValueError("FPS is absent from the PT file; pass --fps explicitly")


def load_gvhmr_pt(path: str | Path, fps: float | None = None) -> tuple[dict[str, np.ndarray], float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to read a GVHMR .pt file") from exc
    try:
        prediction = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced
        prediction = torch.load(path, map_location="cpu")
    if not isinstance(prediction, Mapping):
        raise ValueError(f"GVHMR PT root must be a mapping, got {type(prediction).__name__}")
    if "smpl_params_global" not in prediction:
        raise ValueError("GVHMR PT is missing 'smpl_params_global' (camera-coordinate data is not accepted)")
    params = prediction["smpl_params_global"]
    if not isinstance(params, Mapping):
        raise ValueError("'smpl_params_global' must be a mapping")
    return normalize_smpl_params(params), resolve_fps(prediction, fps)


def write_smpl_csv(path: str | Path, params: Mapping[str, Any], fps: float) -> None:
    normalized = normalize_smpl_params(params)
    fps = _scalar_fps(fps, "fps")
    matrix = np.column_stack(
        (np.full(len(normalized["body_pose"]), fps), normalized["global_orient"],
         normalized["body_pose"], normalized["transl"], normalized["betas"])
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(matrix)


def read_smpl_csv(path: str | Path, expected_fps: float | None = None) -> tuple[dict[str, np.ndarray], float]:
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError("SMPL CSV is empty") from exc
        missing = [name for name in CSV_COLUMNS if name not in columns]
        extra = [name for name in columns if name not in CSV_COLUMNS]
        if missing or extra or columns != CSV_COLUMNS:
            details = []
            if missing:
                details.append(f"missing columns: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected columns: {', '.join(extra)}")
            if not missing and not extra:
                details.append("columns are not in canonical order")
            raise ValueError("invalid SMPL CSV schema (" + "; ".join(details) + ")")
        rows = list(reader)
    if not rows:
        raise ValueError("SMPL CSV contains no frames")
    try:
        matrix = np.asarray(rows, dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"SMPL CSV contains a non-numeric value: {exc}") from exc
    if matrix.shape[1] != 80 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"SMPL CSV must contain 80 finite numeric values per frame, got shape {matrix.shape}")
    fps_values = matrix[:, 0]
    if np.any(fps_values <= 0) or not np.allclose(fps_values, fps_values[0]):
        raise ValueError("SMPL CSV has inconsistent or non-positive FPS values across frames")
    fps = float(fps_values[0])
    if expected_fps is not None and not np.isclose(fps, _scalar_fps(expected_fps, "expected_fps")):
        raise ValueError(f"SMPL CSV FPS {fps} does not match expected FPS {expected_fps}")
    params = normalize_smpl_params({
        "global_orient": matrix[:, 1:4], "body_pose": matrix[:, 4:67],
        "transl": matrix[:, 67:70], "betas": matrix[:, 70:80],
    })
    return params, fps


def as_gmr_smplx_data(params: Mapping[str, Any], fps: float) -> dict[str, Any]:
    normalized = normalize_smpl_params(params)
    return {
        "pose_body": normalized["body_pose"],
        "betas": normalized["betas"],
        "root_orient": normalized["global_orient"],
        "trans": normalized["transl"],
        "mocap_frame_rate": np.asarray(_scalar_fps(fps, "fps"), dtype=np.float32),
    }
