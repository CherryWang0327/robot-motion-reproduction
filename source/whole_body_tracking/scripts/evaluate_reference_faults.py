#!/usr/bin/env python3
"""Evaluate G1 tracking under observation-only temporal reference faults.

The grid is intentionally opt-in.  ``--smoke-test`` is the safe first action.
"""

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent / "rsl_rl"))
import cli_args  # noqa: E402


CHECKPOINT = Path("/home/unitree/projects/whole_body_tracking/logs/rsl_rl/g1_flat/2026-08-05_09-09-29_taichi1_30k_local_resume/model_29999.pt")
MOTION = Path("/home/unitree/projects/whole_body_tracking/inputs/taichi1/taichi1_gmr_50fps.npz")
TASK = "Tracking-Flat-G1-RefFaultEval-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--smoke-test", action="store_true", help="Run only the required three short validation cases.")
parser.add_argument("--run-grid", action="store_true", help="Run the complete 5 x 4 fault grid (explicit opt-in).")
parser.add_argument("--task", default=TASK)
parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT)
parser.add_argument("--motion-file", type=Path, default=MOTION)
parser.add_argument("--output-dir", type=Path, default=Path("results/reference_faults"))
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=None, help="Cap each condition; default evaluates the complete motion.")
parser.add_argument("--freeze-start-step", type=int, default=100, help="100 is 2 seconds at 50 Hz.")
parser.add_argument("--video", action="store_true", help="Record a representative video for every requested condition.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--disable-fabric", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.video:
    args.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import whole_body_tracking.tasks  # noqa: E402,F401


ERRORS = ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot", "error_joint_pos", "error_joint_vel")


def configure(env_cfg, delay: int, freeze: int):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.motion.motion_file = str(args.motion_file.resolve())
    if hasattr(env_cfg.commands.motion, "reference_delay_steps"):
        env_cfg.commands.motion.reference_delay_steps = delay
        env_cfg.commands.motion.reference_freeze_steps = freeze
        env_cfg.commands.motion.reference_freeze_start_step = args.freeze_start_step
    env_cfg.observations.policy.enable_corruption = False
    for field in fields(env_cfg.events):
        setattr(env_cfg.events, field.name, None)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.commands.motion.pose_range = {key: (0.0, 0.0) for key in env_cfg.commands.motion.pose_range}
    env_cfg.commands.motion.velocity_range = {key: (0.0, 0.0) for key in env_cfg.commands.motion.velocity_range}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.episode_length_s = 1.0e9


def place_at_frame_zero(env):
    command = env.unwrapped.command_manager.get_term("motion")
    ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
    command.time_steps.zero_()
    robot = command.robot
    robot.write_joint_state_to_sim(torch.clamp(command.joint_pos, robot.data.soft_joint_pos_limits[:, :, 0], robot.data.soft_joint_pos_limits[:, :, 1]), command.joint_vel, env_ids=ids)
    robot.write_root_state_to_sim(torch.cat([command.body_pos_w[:, 0], command.body_quat_w[:, 0], command.body_lin_vel_w[:, 0], command.body_ang_vel_w[:, 0]], dim=-1), env_ids=ids)
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
    command.time_steps.fill_(-1)
    command._update_command()
    return command


def build_evaluator(env_cfg, agent_cfg):
    """Create Isaac Sim once; its context cannot safely be recreated per grid cell."""
    configure(env_cfg, 10, 0)  # allocate the largest supported delay ring buffer
    torch.manual_seed(args.seed)
    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        # Record one file on every reset.  ``run_condition`` resets once per
        # grid cell, so episode N+1 maps to condition N (episode 0 is setup).
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(args.output_dir.resolve() / "videos"),
            episode_trigger=lambda _: True,
            video_length=args.max_steps or 4002,
            disable_logger=True,
        )
    wrapped = RslRlVecEnvWrapper(env)
    # RslRlVecEnvWrapper resets in __init__.  Teleport only after that reset,
    # otherwise the wrapper overwrites frame zero with an adaptive-sampled frame.
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint_path.resolve()))
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    return wrapped, policy


def run_condition(wrapped, policy, delay: int, freeze: int, out_dir: Path):
    command = wrapped.unwrapped.command_manager.get_term("motion")
    command.cfg.reference_delay_steps = delay
    command.cfg.reference_freeze_steps = freeze
    command.cfg.reference_freeze_start_step = args.freeze_start_step
    # Reset manager state, then overwrite the sampled initial frame with zero.
    wrapped.reset()
    command = place_at_frame_zero(wrapped)
    total = command.motion.time_step_total - 1
    steps = min(total, args.max_steps) if args.max_steps else total
    obs, _ = wrapped.get_observations()
    rows, first_done = [], None
    obs_checks = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for step in range(steps):
        # ``inference_mode`` marks command metrics as immutable inference
        # tensors, which prevents Isaac Lab from clearing them at the next
        # condition reset.  ``no_grad`` is equally inference-only for PPO.
        with torch.no_grad():
            action = policy(obs)
            obs, _, dones, _ = wrapped.step(action)
        policy_frame = int(command.policy_source_time_step[0]) if hasattr(command, "policy_source_time_step") else int(command.time_steps[0])
        row = {"step": step, "clean_reference_frame": int(command.time_steps[0]), "policy_reference_frame": policy_frame}
        row.update({name: float(command.metrics[name][0]) for name in ERRORS})
        rows.append(row)
        # Compact evidence that observation fault differs without touching clean metrics.
        policy_command = command.policy_command[0] if hasattr(command, "policy_command") else command.command[0]
        obs_checks.append({"step": step, "clean_frame": int(command.time_steps[0]), "policy_frame": policy_frame, "policy_command_l2_to_clean": float(torch.linalg.vector_norm(policy_command - command.command[0]))})
        if bool(dones[0]):
            first_done = step
            break
    with (out_dir / "tracking_errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with (out_dir / "observation_fault_trace.json").open("w") as f:
        json.dump(obs_checks, f, indent=2)
    summary = {"delay_steps": delay, "freeze_steps": freeze, "freeze_start_step": args.freeze_start_step, "completed_full_motion": first_done is None and len(rows) == steps, "first_termination_frame": first_done, "mean_episode_length_steps": len(rows), "planned_steps": steps, "metrics": {}}
    for name in ERRORS:
        values = torch.tensor([row[name] for row in rows], dtype=torch.float64)
        summary["metrics"][name] = {"mean": float(values.mean()), "rmse": float(torch.sqrt(torch.mean(values.square())))}
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary, obs_checks


def flat_summary(summary):
    row = {key: summary[key] for key in ("delay_steps", "freeze_steps", "completed_full_motion", "first_termination_frame", "mean_episode_length_steps")}
    row.update({f"{name}_rmse": summary["metrics"][name]["rmse"] for name in ERRORS})
    return row


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if not args.checkpoint_path.is_file() or not args.motion_file.is_file():
        parser.error("checkpoint or motion file does not exist")
    if not args.smoke_test and not args.run_grid:
        parser.error("Refusing to run a grid by default. Use --smoke-test or explicit --run-grid.")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    conditions = [(d, f) for d in (0, 1, 2, 5, 10) for f in (0, 1, 2, 5)] if args.run_grid else [(0, 0), (2, 0), (0, 2)]
    all_rows, traces = [], {}
    wrapped, policy = build_evaluator(env_cfg, agent_cfg)
    for delay, freeze in conditions:
        name = f"delay_{delay:02d}_freeze_{freeze:02d}"
        summary, trace = run_condition(wrapped, policy, delay, freeze, root / name)
        all_rows.append(flat_summary(summary)); traces[name] = trace
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys()); writer.writeheader(); writer.writerows(all_rows)
    protocol = {"task": args.task, "checkpoint": str(args.checkpoint_path.resolve()), "motion": str(args.motion_file.resolve()), "control_hz": 50, "freeze_start_step": args.freeze_start_step, "conditions": [{"delay_steps": d, "freeze_steps": f} for d, f in conditions], "clean_reference_contract": "initialization, rewards, terminations, and metrics use MotionCommand clean properties; only three policy terms are faulted."}
    with (root / "fault_grid.json").open("w") as f:
        json.dump(protocol, f, indent=2)
    # Smoke assertions are intentionally on the policy reference stream. A full
    # baseline-vs-new metric equivalence is reported by delay=0/freeze=0 output.
    if args.smoke_test:
        # At no fault, every policy-reference tensor equals MotionCommand's
        # clean tensor, which is also the tensor used by baseline rewards/metrics.
        no_fault_max_l2 = max(row["policy_command_l2_to_clean"] for row in traces["delay_00_freeze_00"])
        with (root / "smoke_test.json").open("w") as f:
            json.dump({"no_fault_policy_command_max_l2_to_clean": no_fault_max_l2, "equivalence_method": "faulted policy stream equals clean MotionCommand stream at every no-fault smoke step; the original-task comparison is saved separately in baseline_original_task."}, f, indent=2)
        assert all(value["policy_command_l2_to_clean"] < 1e-6 for value in traces["delay_00_freeze_00"])
        delayed = traces["delay_02_freeze_00"]
        assert all(row["policy_frame"] == row["clean_frame"] - 2 for row in delayed[2:])
        frozen = traces["delay_00_freeze_02"]
        start = args.freeze_start_step
        assert frozen[start]["policy_frame"] == frozen[start - 1]["policy_frame"]
        assert frozen[start + 1]["policy_frame"] == frozen[start - 1]["policy_frame"]
        assert frozen[start + 2]["policy_frame"] == frozen[start + 2]["clean_frame"]
    wrapped.close()
    print(f"Wrote {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
