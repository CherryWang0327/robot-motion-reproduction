"""Policy-only observation terms backed by ``ReferenceFaultMotionCommand``."""

from __future__ import annotations

import torch

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .reference_faults import ReferenceFaultMotionCommand


def _command(env, command_name: str) -> ReferenceFaultMotionCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ReferenceFaultMotionCommand):
        raise TypeError("reference-fault observations require ReferenceFaultMotionCommand")
    return command


def faulted_generated_commands(env, command_name: str) -> torch.Tensor:
    return _command(env, command_name).policy_command


def faulted_motion_anchor_pos_b(env, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.policy_anchor_pos_w,
        command.policy_anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)


def faulted_motion_anchor_ori_b(env, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.policy_anchor_pos_w,
        command.policy_anchor_quat_w,
    )
    return matrix_from_quat(ori)[..., :2].reshape(env.num_envs, -1)
