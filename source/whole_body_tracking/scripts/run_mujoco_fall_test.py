#!/usr/bin/env python3
"""Run the teacher MuJoCo deployer headlessly and record objective fall signals."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import mujoco
import numpy as np


DEPLOYER = Path("/home/unitree/BysanRL/BeyondMimic/Beyondmimic_Deploy_G1-main/deploy_mujoco_160d.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--policy_file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_root_height", type=float, default=0.45)
    parser.add_argument("--min_up_z", type=float, default=0.45)
    args = parser.parse_args()
    with np.load(args.motion_file) as motion:
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        frames = int(len(motion["joint_pos"]))
    duration = frames / fps + 2.0
    original_step = mujoco.mj_step
    state = {
        "fell": False,
        "first_fall_sim_time_s": None,
        "minimum_root_height_m": float("inf"),
        "minimum_root_up_z": float("inf"),
        "fall_thresholds": {"root_height_m": args.min_root_height, "root_up_z": args.min_up_z},
    }

    def monitored_step(model, data, *step_args, **step_kwargs):
        result = original_step(model, data, *step_args, **step_kwargs)
        height = float(data.qpos[2])
        w, x, y, z = (float(v) for v in data.qpos[3:7])
        up_z = 1.0 - 2.0 * (x * x + y * y)
        state["minimum_root_height_m"] = min(state["minimum_root_height_m"], height)
        state["minimum_root_up_z"] = min(state["minimum_root_up_z"], up_z)
        if height < args.min_root_height or up_z < args.min_up_z:
            if not state["fell"]:
                state["first_fall_sim_time_s"] = float(data.time)
            state["fell"] = True
        return result

    mujoco.mj_step = monitored_step
    saved_argv = sys.argv[:]
    deploy_dir = str(DEPLOYER.parent)
    inserted_path = deploy_dir not in sys.path
    try:
        if inserted_path:
            sys.path.insert(0, deploy_dir)
        sys.argv = [str(DEPLOYER), "--motion_file", args.motion_file, "--policy_file", args.policy_file,
                    "--duration", str(duration), "--headless"]
        runpy.run_path(str(DEPLOYER), run_name="__main__")
        state["completed"] = True
    except Exception as error:
        state.update(completed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        mujoco.mj_step = original_step
        sys.argv = saved_argv
        if inserted_path and deploy_dir in sys.path:
            sys.path.remove(deploy_dir)
        state.update(motion_file=str(Path(args.motion_file).resolve()),
                     policy_file=str(Path(args.policy_file).resolve()), frames=frames, fps=fps,
                     requested_duration_s=duration)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
