"""Read the project SMPL CSV contract and write an AMASS-compatible NPZ."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _columns(row: dict[str, str], prefixes: tuple[str, ...], size: int) -> np.ndarray:
    """Read either ``name_0`` or ``name0`` numbered CSV columns."""
    for prefix in prefixes:
        values = []
        for index in range(size):
            key = f"{prefix}_{index}" if f"{prefix}_{index}" in row else f"{prefix}{index}"
            if key not in row:
                break
            values.append(float(row[key]))
        if len(values) == size:
            return np.asarray(values, dtype=np.float32)
    raise ValueError(f"Missing {size} columns for one of: {', '.join(prefixes)}")


def csv_to_amass(csv_path: Path, output_path: Path) -> None:
    """Convert 80-column SMPL CSV to the minimal AMASS fields ProtoMotions reads.

    The conversion deliberately preserves the GVHMR gravity-aligned Y-up values.
    ``convert_amass_to_proto.py --source-y-up`` owns the one recorded Y-up to
    Z-up conversion further downstream.
    """
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"SMPL CSV has no frames: {csv_path}")

    poses = np.stack([
        np.concatenate((_columns(row, ("global_orient", "root_orient"), 3), _columns(row, ("body_pose",), 63), np.zeros(6, dtype=np.float32)))
        for row in rows
    ])
    trans = np.stack([_columns(row, ("trans", "transl"), 3) for row in rows])
    betas = _columns(rows[0], ("betas",), 10)
    fps_values = [float(row.get("fps", row.get("mocap_framerate", "30"))) for row in rows]
    if not np.allclose(fps_values, fps_values[0]):
        raise ValueError("SMPL CSV fps must be constant across frames")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, poses=poses, trans=trans, betas=betas, gender="neutral", mocap_framerate=float(fps_values[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the standard SMPL CSV contract to AMASS NPZ")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    csv_to_amass(args.input, args.output)


if __name__ == "__main__":
    main()
