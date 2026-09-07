from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch


def convert_gmr_pkl_to_pt(source: Path, output: Path) -> None:
    """Adapt a GMR PKL to the PT contract already consumed by WBT's converter.

    GMR's root quaternion is an xyzw quaternion.  The output keeps that order;
    WBT's ``pt_to_npz_local.py`` is explicitly invoked with the same contract.
    """
    with source.open("rb") as stream:
        data = pickle.load(stream)
    required = ("root_pos", "root_rot", "dof_pos", "fps")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("GMR PKL missing fields: " + ", ".join(missing))
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must be (T, 3), got {root_pos.shape}")
    if root_rot.shape != (root_pos.shape[0], 4):
        raise ValueError(f"root_rot must be (T, 4), got {root_rot.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[0] != root_pos.shape[0] or dof_pos.shape[1] != 29:
        raise ValueError(f"dof_pos must be (T, 29), got {dof_pos.shape}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "gts": torch.from_numpy(root_pos[:, None, :]),
        "grs": torch.from_numpy(root_rot[:, None, :]),
        "dps": torch.from_numpy(dof_pos),
        "motion_dt": torch.tensor(1.0 / fps, dtype=torch.float32),
    }, output)
