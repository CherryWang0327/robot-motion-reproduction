#!/usr/bin/env python3
"""Train ProtoMotions/PyRoki references with the unchanged WBT baseline protocol."""

from pathlib import Path

import run_ten_motion_baseline as runner


ROOT = Path(__file__).resolve().parents[1]
runner.MOTIONS = {
    "proto_happy": ROOT
    / "retarget_comparison/protomotions/happy/04_wbt_npz/happy_smpl_50fps_pyroki_proto.npz",
    "proto_taichi1": ROOT
    / "retarget_comparison/protomotions/taichi1/04_wbt_npz/taichi1_smpl_50fps_pyroki_proto.npz",
}
runner.RESULT_ROOT = ROOT / "results" / "retarget_comparison"


if __name__ == "__main__":
    runner.main()
