"""Evaluate or play a checkpoint from RSL-RL on one local reference motion."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RSL-RL motion-tracking policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record the evaluation video.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Video/evaluation length in control steps. Defaults to one complete reference motion.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of evaluation environments (default: 1).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, required=True, help="Absolute path to the local motion NPZ file.")
parser.add_argument(
    "--evaluation_output",
    type=str,
    default=None,
    help="Evaluation output directory. Defaults to <checkpoint_dir>/evaluation/<motion_name>.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import json
import os
import pathlib
from dataclasses import fields

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_error_magnitude, quat_inv, quat_mul, yaw_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Run a deterministic, full-motion evaluation of an RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    motion_file = pathlib.Path(args_cli.motion_file).expanduser().resolve()
    if not motion_file.is_file():
        parser.error(f"--motion_file does not exist or is not a file: {motion_file}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Use the explicitly selected motion and remove stochastic evaluation effects.
    env_cfg.commands.motion.motion_file = str(motion_file)
    env_cfg.observations.policy.enable_corruption = False
    # Keep a valid EventCfg object (ManagerBase expects one), but disable every
    # randomization term for deterministic evaluation.
    for event_field in fields(env_cfg.events):
        setattr(env_cfg.events, event_field.name, None)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.commands.motion.pose_range = {key: (0.0, 0.0) for key in env_cfg.commands.motion.pose_range}
    env_cfg.commands.motion.velocity_range = {key: (0.0, 0.0) for key in env_cfg.commands.motion.velocity_range}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.episode_length_s = 1.0e9
    print(f"[INFO]: Evaluating local motion: {motion_file}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)
    motion_command = env.unwrapped.command_manager.get_term("motion")
    # The environment advances the command before returning from env.step().
    # Starting exactly at frame 0 therefore yields valid evaluated transitions
    # for reference frames 1..T-1. A T-th step would wrap/resample the command
    # and contaminate the final row with an episode-boundary discontinuity.
    full_motion_transition_steps = max(motion_command.motion.time_step_total - 1, 0)
    evaluation_steps = args_cli.video_length or full_motion_transition_steps
    if evaluation_steps > full_motion_transition_steps:
        parser.error(
            f"--video_length={evaluation_steps} exceeds the {full_motion_transition_steps} "
            "non-wrapping transitions available in this motion"
        )
    evaluation_dir = pathlib.Path(args_cli.evaluation_output).expanduser() if args_cli.evaluation_output else (
        pathlib.Path(log_dir) / "evaluation" / motion_file.stem
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(evaluation_dir / "video"),
            "step_trigger": lambda step: step == 0,
            "video_length": evaluation_steps,
            "disable_logger": True,
        }
        print("[INFO] Recording one complete evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    # Start every environment at reference frame zero. Random pose/velocity offsets
    # were disabled above, so this also restores the exact reference initial state.
    env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
    motion_command.time_steps.zero_()
    joint_pos = torch.clamp(
        motion_command.joint_pos,
        motion_command.robot.data.soft_joint_pos_limits[:, :, 0],
        motion_command.robot.data.soft_joint_pos_limits[:, :, 1],
    )
    motion_command.robot.write_joint_state_to_sim(joint_pos, motion_command.joint_vel, env_ids=env_ids)
    motion_command.robot.write_root_state_to_sim(
        torch.cat(
            [
                motion_command.body_pos_w[:, 0],
                motion_command.body_quat_w[:, 0],
                motion_command.body_lin_vel_w[:, 0],
                motion_command.body_ang_vel_w[:, 0],
            ],
            dim=-1,
        ),
        env_ids=env_ids,
    )
    # Refresh derived link states after teleporting the articulation. Without
    # this, the first termination check sees stale body transforms.
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
    # The environment's initial reset sampled a random command frame. Rebuild
    # relative body targets for frame zero before the first termination check.
    motion_command.time_steps.fill_(-1)
    motion_command._update_command()

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    error_names = (
        "error_anchor_pos",
        "error_anchor_rot",
        "error_body_pos",
        "error_body_rot",
        "error_joint_pos",
        "error_joint_vel",
    )
    error_rows = []
    detailed_rows = []
    first_done_step = None
    termination_reasons = []
    joint_names = motion_command.robot.joint_names
    body_names = motion_command.cfg.body_names
    foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_body_indexes = [body_names.index(name) for name in foot_names]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    foot_sensor_indexes = contact_sensor.find_bodies(foot_names, preserve_order=True)[0]
    previous_foot_pos_xy = None
    accumulated_foot_slip = torch.zeros(len(foot_names), dtype=torch.float64)
    previous_action = None
    # simulate environment
    while simulation_app.is_running() and timestep < evaluation_steps:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
        row = {"step": timestep, "reference_frame": int(motion_command.time_steps[0].item())}
        row.update({name: float(motion_command.metrics[name][0].item()) for name in error_names})
        error_rows.append(row)

        anchor_delta = motion_command.anchor_pos_w[0] - motion_command.robot_anchor_pos_w[0]
        joint_pos_delta = motion_command.joint_pos[0] - motion_command.robot_joint_pos[0]
        joint_vel_delta = motion_command.joint_vel[0] - motion_command.robot_joint_vel[0]
        body_pos_error = torch.norm(
            motion_command.body_pos_relative_w[0] - motion_command.robot_body_pos_w[0], dim=-1
        )
        body_rot_error = quat_error_magnitude(
            motion_command.body_quat_relative_w[0], motion_command.robot_body_quat_w[0]
        )
        detailed = dict(row)
        action_vector = actions[0].detach()
        detailed["action_norm"] = float(torch.linalg.vector_norm(action_vector).item())
        if previous_action is None:
            detailed["action_delta_norm"] = 0.0
        else:
            detailed["action_delta_norm"] = float(
                torch.linalg.vector_norm(action_vector - previous_action).item()
            )
        previous_action = action_vector.clone()
        anchor_yaw_delta = quat_mul(
            yaw_quat(motion_command.robot_anchor_quat_w[0:1]),
            quat_inv(yaw_quat(motion_command.anchor_quat_w[0:1])),
        )[0]
        raw_yaw_error = 2.0 * torch.atan2(anchor_yaw_delta[3], anchor_yaw_delta[0])
        detailed["root_yaw_error_rad"] = float(
            torch.atan2(torch.sin(raw_yaw_error), torch.cos(raw_yaw_error)).item()
        )
        for axis, value in zip("xyz", anchor_delta, strict=True):
            detailed[f"anchor_pos_error_{axis}"] = float(value.item())
        for index, name in enumerate(joint_names):
            detailed[f"joint_pos_error.{name}"] = float(joint_pos_delta[index].item())
            detailed[f"joint_vel_error.{name}"] = float(joint_vel_delta[index].item())
            detailed[f"joint_position.{name}"] = float(motion_command.robot_joint_pos[0, index].item())
            detailed[f"joint_torque.{name}"] = float(motion_command.robot.data.applied_torque[0, index].item())
        for index, name in enumerate(body_names):
            detailed[f"body_pos_error.{name}"] = float(body_pos_error[index].item())
            detailed[f"body_rot_error.{name}"] = float(body_rot_error[index].item())

        foot_pos_xy = motion_command.robot_body_pos_w[0, foot_body_indexes, :2].detach().cpu()
        reference_foot_pos_xy = motion_command.body_pos_relative_w[0, foot_body_indexes, :2].detach().cpu()
        foot_forces = torch.norm(contact_sensor.data.net_forces_w[0, foot_sensor_indexes], dim=-1).detach().cpu()
        foot_contacts = foot_forces > contact_sensor.cfg.force_threshold
        if previous_foot_pos_xy is None:
            foot_slip_step = torch.zeros(len(foot_names), dtype=torch.float64)
        else:
            foot_slip_step = torch.norm(foot_pos_xy - previous_foot_pos_xy, dim=-1).to(torch.float64)
            foot_slip_step *= foot_contacts.to(torch.float64)
            accumulated_foot_slip += foot_slip_step
        previous_foot_pos_xy = foot_pos_xy
        for index, name in enumerate(foot_names):
            detailed[f"contact.{name}"] = int(foot_contacts[index].item())
            detailed[f"contact_force.{name}"] = float(foot_forces[index].item())
            detailed[f"foot_slip_step.{name}"] = float(foot_slip_step[index].item())
            detailed[f"foot_pos_x.{name}"] = float(foot_pos_xy[index, 0].item())
            detailed[f"foot_pos_y.{name}"] = float(foot_pos_xy[index, 1].item())
            detailed[f"foot_pos_z.{name}"] = float(
                motion_command.robot_body_pos_w[0, foot_body_indexes[index], 2].item()
            )
            detailed[f"reference_foot_pos_x.{name}"] = float(reference_foot_pos_xy[index, 0].item())
            detailed[f"reference_foot_pos_y.{name}"] = float(reference_foot_pos_xy[index, 1].item())
            detailed[f"reference_foot_pos_z.{name}"] = float(
                motion_command.body_pos_relative_w[0, foot_body_indexes[index], 2].item()
            )

        limits = motion_command.robot.data.soft_joint_pos_limits[0]
        limit_margin = torch.minimum(
            motion_command.robot_joint_pos[0] - limits[:, 0], limits[:, 1] - motion_command.robot_joint_pos[0]
        )
        detailed["minimum_joint_limit_margin"] = float(limit_margin.min().item())
        detailed["absolute_joint_power"] = float(
            torch.sum(torch.abs(motion_command.robot.data.applied_torque[0] * motion_command.robot_joint_vel[0])).item()
        )
        detailed_rows.append(detailed)
        if first_done_step is None and bool(dones[0].item()):
            first_done_step = timestep
            term_dones = getattr(env.unwrapped.termination_manager, "_term_dones", {})
            termination_reasons = [
                name for name, term_done in term_dones.items() if bool(term_done[0].item())
            ]
        timestep += 1
        if first_done_step is not None:
            break

    csv_path = evaluation_dir / "tracking_errors.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=error_rows[0].keys())
        writer.writeheader()
        writer.writerows(error_rows)

    detailed_csv_path = evaluation_dir / "tracking_errors_detailed.csv"
    with detailed_csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=detailed_rows[0].keys())
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary = {
        "motion_file": str(motion_file),
        "checkpoint": str(pathlib.Path(resume_path).resolve()),
        "num_envs": env.unwrapped.num_envs,
        "planned_steps": evaluation_steps,
        "evaluated_reference_frame_start": 1 if evaluation_steps else None,
        "evaluated_reference_frame_end": evaluation_steps if evaluation_steps else None,
        "reference_total_frames": motion_command.motion.time_step_total,
        "signed_anchor_error_convention": "reference_minus_policy",
        "recorded_steps": len(error_rows),
        "completed_without_termination": first_done_step is None,
        "first_termination_step": first_done_step,
        "termination_reasons": termination_reasons,
        "metrics": {},
    }
    for name in error_names:
        values = torch.tensor([row[name] for row in error_rows], dtype=torch.float64)
        summary["metrics"][name] = {
            "mean": float(values.mean()),
            "rmse": float(torch.sqrt(torch.mean(values.square()))),
            "p95": float(torch.quantile(values, 0.95)),
            "max": float(values.max()),
        }
    def statistics(values: torch.Tensor) -> dict:
        values = values.to(torch.float64)
        absolute = values.abs()
        return {
            "mean_absolute": float(absolute.mean()),
            "rmse": float(torch.sqrt(torch.mean(values.square()))),
            "p95_absolute": float(torch.quantile(absolute, 0.95)),
            "max_absolute": float(absolute.max()),
        }

    summary["detailed_metrics"] = {
        "anchor_position_axes": {},
        "joints": {},
        "bodies": {},
        "feet": {},
        "energy": {},
        "joint_limits": {},
        "control": {},
    }
    for axis in "xyz":
        values = torch.tensor([row[f"anchor_pos_error_{axis}"] for row in detailed_rows])
        summary["detailed_metrics"]["anchor_position_axes"][axis] = statistics(values)
    for name in joint_names:
        pos_values = torch.tensor([row[f"joint_pos_error.{name}"] for row in detailed_rows])
        vel_values = torch.tensor([row[f"joint_vel_error.{name}"] for row in detailed_rows])
        summary["detailed_metrics"]["joints"][name] = {
            "position_error": statistics(pos_values),
            "velocity_error": statistics(vel_values),
        }
    for name in body_names:
        pos_values = torch.tensor([row[f"body_pos_error.{name}"] for row in detailed_rows])
        rot_values = torch.tensor([row[f"body_rot_error.{name}"] for row in detailed_rows])
        summary["detailed_metrics"]["bodies"][name] = {
            "position_error": statistics(pos_values),
            "rotation_error": statistics(rot_values),
        }
    for index, name in enumerate(foot_names):
        contacts = torch.tensor([row[f"contact.{name}"] for row in detailed_rows], dtype=torch.float64)
        forces = torch.tensor([row[f"contact_force.{name}"] for row in detailed_rows])
        slip_speeds = torch.tensor(
            [row[f"foot_slip_step.{name}"] / env.unwrapped.step_dt for row in detailed_rows]
        )
        contact_slip_speeds = slip_speeds[contacts.bool()]
        summary["detailed_metrics"]["feet"][name] = {
            "contact_fraction": float(contacts.mean()),
            "contact_force_max": float(forces.max()),
            "slip_distance_during_contact": float(accumulated_foot_slip[index]),
            "slip_speed_during_contact": (
                statistics(contact_slip_speeds) if contact_slip_speeds.numel() else None
            ),
        }
    power = torch.tensor([row["absolute_joint_power"] for row in detailed_rows])
    margins = torch.tensor([row["minimum_joint_limit_margin"] for row in detailed_rows])
    summary["detailed_metrics"]["energy"] = {
        "mean_absolute_joint_power": float(power.mean()),
        "absolute_joint_energy": float(power.sum() * env.unwrapped.step_dt),
    }
    summary["detailed_metrics"]["joint_limits"] = {
        "minimum_soft_limit_margin": float(margins.min()),
        "steps_outside_soft_limits": int((margins < 0.0).sum()),
    }
    action_norm = torch.tensor([row["action_norm"] for row in detailed_rows])
    action_delta_norm = torch.tensor([row["action_delta_norm"] for row in detailed_rows[1:]])
    root_yaw_error = torch.tensor([row["root_yaw_error_rad"] for row in detailed_rows])
    summary["detailed_metrics"]["control"] = {
        "action_norm": statistics(action_norm),
        "action_delta_norm": statistics(action_delta_norm) if action_delta_norm.numel() else None,
        "root_yaw_error_rad": statistics(root_yaw_error),
    }
    summary_path = evaluation_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[INFO] Tracking errors: {csv_path.resolve()}")
    print(f"[INFO] Detailed tracking data: {detailed_csv_path.resolve()}")
    print(f"[INFO] Evaluation summary: {summary_path.resolve()}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
