#!/usr/bin/env python3
"""Evaluate a 160D ONNX policy under reproducible MuJoCo perturbations.

The deployed ONNX, source motion NPZ, and MuJoCo XML are never modified.  The
requested friction, mass, and pelvis-push changes exist only in this process.
"""

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
    parser.add_argument("--motion-file", required=True, type=Path)
    parser.add_argument("--policy-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--friction-scale", type=float, default=1.0)
    parser.add_argument("--body-mass-scale", type=float, default=1.0)
    parser.add_argument("--push-start-s", type=float, default=-1.0)
    parser.add_argument("--push-duration-s", type=float, default=0.20)
    parser.add_argument("--push-force-x", type=float, default=0.0)
    parser.add_argument("--push-force-y", type=float, default=0.0)
    parser.add_argument("--push-force-z", type=float, default=0.0)
    parser.add_argument("--min-root-height", type=float, default=0.45)
    parser.add_argument("--min-up-z", type=float, default=0.45)
    args = parser.parse_args()

    if not args.motion_file.is_file():
        parser.error(f"--motion-file does not exist: {args.motion_file}")
    if not args.policy_file.is_file():
        parser.error(f"--policy-file does not exist: {args.policy_file}")
    if not DEPLOYER.is_file():
        parser.error(f"MuJoCo deployer does not exist: {DEPLOYER}")
    if args.friction_scale <= 0.0 or args.body_mass_scale <= 0.0:
        parser.error("friction and body-mass scales must be positive")
    if args.push_duration_s <= 0.0:
        parser.error("--push-duration-s must be positive")

    with np.load(args.motion_file) as motion:
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        frames = int(len(motion["joint_pos"]))
    duration_s = frames / fps + 2.0
    push_force = np.array(
        [args.push_force_x, args.push_force_y, args.push_force_z], dtype=np.float64
    )
    state: dict[str, object] = {
        "completed": False,
        "fell": False,
        "first_fall_sim_time_s": None,
        "minimum_root_height_m": float("inf"),
        "minimum_root_up_z": float("inf"),
        "fall_thresholds": {
            "root_height_m": args.min_root_height,
            "root_up_z": args.min_up_z,
        },
        "perturbations": {
            "friction_scale": args.friction_scale,
            "body_mass_scale": args.body_mass_scale,
            "push_start_s": args.push_start_s,
            "push_duration_s": args.push_duration_s,
            "push_force_world_n": push_force.tolist(),
        },
        "push_applied": False,
    }

    original_step = mujoco.mj_step
    original_model_loader = mujoco.MjModel.from_xml_path
    push_body_id: int | None = None

    def load_perturbed_model(*loader_args, **loader_kwargs):
        nonlocal push_body_id
        model = original_model_loader(*loader_args, **loader_kwargs)
        model.geom_friction[:] *= args.friction_scale
        model.body_mass[1:] *= args.body_mass_scale
        model.body_inertia[1:] *= args.body_mass_scale
        push_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if push_body_id < 0:
            raise RuntimeError("MuJoCo model has no pelvis body for the push test")
        return model

    def monitored_step(model, data, *step_args, **step_kwargs):
        active_push = (
            args.push_start_s >= 0.0
            and args.push_start_s <= data.time < args.push_start_s + args.push_duration_s
        )
        if active_push:
            data.xfrc_applied[push_body_id, :3] = push_force
            state["push_applied"] = True
        result = original_step(model, data, *step_args, **step_kwargs)
        if active_push:
            data.xfrc_applied[push_body_id, :] = 0.0

        height = float(data.qpos[2])
        _, x, y, _ = (float(value) for value in data.qpos[3:7])
        up_z = 1.0 - 2.0 * (x * x + y * y)
        state["minimum_root_height_m"] = min(float(state["minimum_root_height_m"]), height)
        state["minimum_root_up_z"] = min(float(state["minimum_root_up_z"]), up_z)
        if height < args.min_root_height or up_z < args.min_up_z:
            if not state["fell"]:
                state["first_fall_sim_time_s"] = float(data.time)
            state["fell"] = True
        return result

    saved_argv = sys.argv[:]
    deploy_dir = str(DEPLOYER.parent)
    inserted_path = deploy_dir not in sys.path
    mujoco.MjModel.from_xml_path = load_perturbed_model
    mujoco.mj_step = monitored_step
    try:
        if inserted_path:
            sys.path.insert(0, deploy_dir)
        sys.argv = [
            str(DEPLOYER), "--motion_file", str(args.motion_file), "--policy_file",
            str(args.policy_file), "--duration", str(duration_s), "--headless",
        ]
        runpy.run_path(str(DEPLOYER), run_name="__main__")
        state["completed"] = True
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        mujoco.mj_step = original_step
        mujoco.MjModel.from_xml_path = original_model_loader
        sys.argv = saved_argv
        if inserted_path and deploy_dir in sys.path:
            sys.path.remove(deploy_dir)
        state.update(
            motion_file=str(args.motion_file.resolve()),
            policy_file=str(args.policy_file.resolve()),
            frames=frames,
            fps=fps,
            requested_duration_s=duration_s,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
