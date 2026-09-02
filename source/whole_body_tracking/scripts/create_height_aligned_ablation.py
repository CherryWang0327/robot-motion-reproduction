#!/usr/bin/env python3
"""Create provenance-preserving rigid-Z PyRoki reference ablations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MOTIONS = ("happy", "taichi1")
# Full G1 body order in the WBT NPZ, verified against the Isaac articulation.
FOOT_INDICES = (18, 19)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest = {"method": "rigid_z_shift_matching_minimum_ankle_height", "motions": {}}
    for motion in MOTIONS:
        gmr_path = ROOT / f"inputs/{motion}/{motion}_gmr_50fps.npz"
        source_path = ROOT / (
            f"retarget_comparison/protomotions/{motion}/04_wbt_npz/"
            f"{motion}_smpl_50fps_pyroki_proto.npz"
        )
        output_dir = ROOT / f"retarget_comparison/height_aligned/{motion}"
        output_path = output_dir / f"{motion}_pyroki_height_aligned_50fps.npz"
        output_dir.mkdir(parents=True, exist_ok=True)
        with np.load(gmr_path) as data:
            gmr_foot_min = float(np.asarray(data["body_pos_w"])[:, FOOT_INDICES, 2].min())
        with np.load(source_path) as data:
            arrays = {key: np.asarray(data[key]).copy() for key in data.files}
        proto_foot_min = float(arrays["body_pos_w"][:, FOOT_INDICES, 2].min())
        shift = gmr_foot_min - proto_foot_min
        arrays["body_pos_w"][:, :, 2] += shift
        np.savez_compressed(output_path, **arrays)
        manifest["motions"][motion] = {
            "gmr_path": str(gmr_path.resolve()),
            "gmr_sha256": digest(gmr_path),
            "source_pyroki_path": str(source_path.resolve()),
            "source_pyroki_sha256": digest(source_path),
            "output_path": str(output_path.resolve()),
            "output_sha256": digest(output_path),
            "gmr_min_ankle_z_m": gmr_foot_min,
            "pyroki_min_ankle_z_before_m": proto_foot_min,
            "applied_rigid_z_shift_m": shift,
            "changed_array": "body_pos_w[..., 2] only",
        }
    manifest_path = ROOT / "results/height_aligned_ablation/reference_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
