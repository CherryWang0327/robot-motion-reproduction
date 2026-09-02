#!/usr/bin/env python3
"""Export GVHMR world-coordinate SMPL parameters to the canonical 80-column CSV."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


PREFIX = "smpl_params_global_"
CSV_COLUMNS = (
    ["fps"]
    + [f"{PREFIX}global_orient_{i}" for i in range(3)]
    + [f"{PREFIX}body_pose_{i}" for i in range(63)]
    + [f"{PREFIX}transl_{i}" for i in range(3)]
    + [f"{PREFIX}betas_{i}" for i in range(10)]
)


def as_numpy(value, name):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def normalize(params):
    required = ("global_orient", "body_pose", "transl", "betas")
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(f"smpl_params_global is missing required field(s): {', '.join(missing)}")
    orient = as_numpy(params["global_orient"], "global_orient")
    pose = as_numpy(params["body_pose"], "body_pose")
    transl = as_numpy(params["transl"], "transl")
    betas = as_numpy(params["betas"], "betas")
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
    for name, array in (("body_pose", pose), ("transl", transl)):
        if array.shape[0] != frames:
            raise ValueError(
                f"frame count mismatch: global_orient has {frames} frames but {name} has {array.shape[0]}"
            )
    if betas.shape in ((10,), (1, 10)):
        betas = np.broadcast_to(betas.reshape(1, 10), (frames, 10))
    elif betas.shape != (frames, 10):
        raise ValueError(f"betas must have shape [10], [1, 10], or [T, 10], got {betas.shape}")
    return orient, pose, transl, betas


def scalar_fps(value, source):
    array = as_numpy(value, source).reshape(-1)
    if array.size != 1 or float(array[0]) <= 0:
        raise ValueError(f"{source} must be one positive FPS value")
    return float(array[0])


def resolve_fps(prediction, explicit):
    metadata = [(key, scalar_fps(prediction[key], key))
                for key in ("fps", "video_fps", "mocap_frame_rate") if key in prediction]
    if metadata and any(not np.isclose(value, metadata[0][1]) for _, value in metadata[1:]):
        raise ValueError(f"inconsistent FPS metadata: {metadata}")
    if explicit is not None:
        fps = scalar_fps(explicit, "--fps")
        if metadata and not np.isclose(fps, metadata[0][1]):
            raise ValueError(f"--fps {fps} conflicts with {metadata[0][0]}={metadata[0][1]} in the PT file")
        return fps
    if metadata:
        return metadata[0][1]
    raise ValueError("FPS is absent from hmr4d_results.pt; pass --fps explicitly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="GVHMR hmr4d_results.pt")
    parser.add_argument("--output", required=True, help="Canonical 80-column SMPL CSV")
    parser.add_argument("--fps", type=float, help="Required unless reliable FPS metadata exists in the PT")
    args = parser.parse_args()
    try:
        prediction = torch.load(args.input, map_location="cpu", weights_only=False)
    except TypeError:
        prediction = torch.load(args.input, map_location="cpu")
    if not isinstance(prediction, dict) or "smpl_params_global" not in prediction:
        raise ValueError("input must contain 'smpl_params_global'; camera-coordinate data is not accepted")
    orient, pose, transl, betas = normalize(prediction["smpl_params_global"])
    fps = resolve_fps(prediction, args.fps)
    matrix = np.column_stack((np.full(orient.shape[0], fps), orient, pose, transl, betas))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(matrix)
    print(f"Saved {orient.shape[0]} frames at {fps:g} FPS to {output}")


if __name__ == "__main__":
    main()
