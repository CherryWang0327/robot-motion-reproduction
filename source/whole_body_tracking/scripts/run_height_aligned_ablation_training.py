#!/usr/bin/env python3
"""Train rigid-Z PyRoki ablations with the unchanged WBT baseline protocol."""

from pathlib import Path

import run_ten_motion_baseline as runner


ROOT = Path(__file__).resolve().parents[1]
runner.MOTIONS = {
    name: ROOT / f"retarget_comparison/height_aligned/{name.removeprefix('proto_zalign_')}/{name.removeprefix('proto_zalign_')}_pyroki_height_aligned_50fps.npz"
    for name in ("proto_zalign_happy", "proto_zalign_taichi1")
}
runner.RESULT_ROOT = ROOT / "results/height_aligned_ablation"


if __name__ == "__main__":
    runner.main()
