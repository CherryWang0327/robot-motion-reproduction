#!/usr/bin/env python3
"""Rebuild tutorial-compatible Proto references from projects/ and launch WBT."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/unitree/miniconda3/envs/whole_body_tracking/bin/python")
NPZS = [
    ROOT / "retarget_comparison/protomotions/happy/04_wbt_npz/happy_smpl_50fps_pyroki_proto.npz",
    ROOT / "retarget_comparison/protomotions/taichi1/04_wbt_npz/taichi1_smpl_50fps_pyroki_proto.npz",
]


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("[STRICT] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)
    run(
        [
            str(PYTHON),
            "scripts/run_protomotions_comparison_pipeline.py",
            "--motions", "happy", "taichi1",
        ],
        env=clean_env,
    )
    run(
        [
            str(PYTHON),
            "scripts/validate_motion_inputs.py",
            *(str(path) for path in NPZS),
            "--output",
            "results/retarget_comparison/proto_input_validation.json",
        ]
    )
    run(
        [
            str(PYTHON),
            "scripts/run_retarget_comparison_training.py",
            "--num_envs", "4096",
            "--min_iterations", "30000",
            "--max_iterations", "100000",
            "--window", "1000",
            "--poll_seconds", "60",
        ]
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[STRICT ERROR] stage failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
