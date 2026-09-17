# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--resume_checkpoint",
    type=str,
    default=None,
    help="Absolute checkpoint path to resume toward --max_iterations as a total iteration budget.",
)
parser.add_argument(
    "--resume_log_dir",
    type=str,
    default=None,
    help="Existing absolute run directory in which resumed TensorBoard events/checkpoints are appended.",
)
motion_source = parser.add_mutually_exclusive_group()
motion_source.add_argument("--registry_name", type=str, help="The name of the W&B registry artifact.")
motion_source.add_argument("--motion_file", type=str, help="Path to a local motion NPZ file (absolute or relative to cwd).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.registry_name is None and args_cli.motion_file is None:
    parser.error("one of --registry_name or --motion_file is required")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Resolve exactly one motion source. Local mode never imports or contacts W&B.
    registry_name = args_cli.registry_name
    if args_cli.motion_file is not None:
        motion_file = pathlib.Path(args_cli.motion_file).expanduser()
        if not motion_file.is_absolute():
            motion_file = pathlib.Path.cwd() / motion_file
        motion_file = motion_file.resolve()
        if not motion_file.is_file():
            parser.error(f"--motion_file does not exist or is not a file: {motion_file}")
        env_cfg.commands.motion.motion_file = str(motion_file)
        print(f"[INFO] Using local motion file: {motion_file}")
    else:
        if ":" not in registry_name:  # append default alias when omitted
            registry_name += ":latest"
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    if args_cli.resume_log_dir is not None:
        resumed_log_dir = pathlib.Path(args_cli.resume_log_dir).expanduser().resolve()
        if not resumed_log_dir.is_dir():
            parser.error(f"--resume_log_dir does not exist or is not a directory: {resumed_log_dir}")
        log_dir = str(resumed_log_dir)
        print(f"[INFO] Appending resumed logs and checkpoints to: {log_dir}")
    else:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if args_cli.resume_checkpoint is not None:
        resume_path = pathlib.Path(args_cli.resume_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
            parser.error(f"--resume_checkpoint does not exist or is not a file: {resume_path}")
        print(f"[INFO]: Resuming from explicit checkpoint: {resume_path}")
        runner.load(str(resume_path))
        # Late-stage checkpoints can contain an unstable directly-optimized
        # action-noise scalar.  Keep the recovered exploration scale fixed so
        # PPO can continue optimizing the actor and critic without NaN scales.
        if hasattr(runner.alg.policy, "std"):
            runner.alg.policy.std.requires_grad_(False)
            print("[INFO]: Frozen resumed action-noise standard deviation.")
    elif agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    learning_iterations = agent_cfg.max_iterations
    if args_cli.resume_checkpoint is not None:
        learning_iterations = max(agent_cfg.max_iterations - runner.current_learning_iteration, 0)
        print(
            f"[INFO]: Checkpoint iteration {runner.current_learning_iteration}; "
            f"training {learning_iterations} remaining iterations toward total {agent_cfg.max_iterations}."
        )
    runner.learn(num_learning_iterations=learning_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
