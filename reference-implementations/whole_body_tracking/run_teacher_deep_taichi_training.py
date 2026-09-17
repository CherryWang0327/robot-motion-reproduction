#!/usr/bin/env python3
"""Teacher-requested deep-network WBT ablation for PyRoki taichi1."""

from pathlib import Path

import run_ten_motion_baseline as runner


ROOT = Path(__file__).resolve().parents[1]
runner.MOTIONS = {
    "proto_taichi1_deep": ROOT
    / "retarget_comparison/protomotions/taichi1/04_wbt_npz/taichi1_smpl_50fps_pyroki_proto.npz"
}
runner.RESULT_ROOT = ROOT / "results" / "teacher_deep_taichi"


if __name__ == "__main__":
    runner.main()

