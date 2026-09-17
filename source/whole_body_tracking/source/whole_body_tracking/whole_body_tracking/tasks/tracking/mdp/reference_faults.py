"""Observation-only temporal faults for deterministic reference evaluation.

The clean :class:`MotionCommand` remains the sole source for dynamics, rewards,
terminations, metrics, and initialization.  This module keeps a separate
per-environment history solely for the three policy observation terms.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

from .commands import MotionCommand, MotionCommandCfg


class ReferenceFaultMotionCommand(MotionCommand):
    """Motion command with clean state plus a faulted policy-reference stream."""

    cfg: "ReferenceFaultMotionCommandCfg"

    def __init__(self, cfg: "ReferenceFaultMotionCommandCfg", env):
        super().__init__(cfg, env)
        if cfg.reference_delay_steps < 0 or cfg.reference_freeze_steps < 0:
            raise ValueError("reference delay and freeze lengths must be non-negative")
        self._history_size = max(int(cfg.reference_delay_steps) + 2, 2)
        joint_dim = self.motion.joint_pos.shape[1]
        self._command_history = torch.zeros(self.num_envs, self._history_size, 2 * joint_dim, device=self.device)
        self._anchor_pos_history = torch.zeros(self.num_envs, self._history_size, 3, device=self.device)
        self._anchor_quat_history = torch.zeros(self.num_envs, self._history_size, 4, device=self.device)
        self._history_cursor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_reference_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_clean_step = torch.full((self.num_envs,), -2, dtype=torch.long, device=self.device)
        self._policy_command = torch.zeros(self.num_envs, 2 * joint_dim, device=self.device)
        self._policy_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._policy_anchor_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._policy_source_time_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._frozen_command = self._policy_command.clone()
        self._frozen_anchor_pos_w = self._policy_anchor_pos_w.clone()
        self._frozen_anchor_quat_w = self._policy_anchor_quat_w.clone()
        self._frozen_source_time_step = self._policy_source_time_step.clone()

    def _resample_command(self, env_ids: Sequence[int]):
        # Parent behavior deliberately stays intact: it selects clean frames and
        # initializes the robot from those clean targets.
        super()._resample_command(env_ids)
        if not hasattr(self, "_last_clean_step"):
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel():
            self._bootstrap_policy_history(ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # CommandTerm.reset invokes our _resample_command; this override makes
        # partial resets explicit and robust if a caller supplies a slice.
        return super().reset(env_ids)

    def _bootstrap_policy_history(self, ids: torch.Tensor) -> None:
        clean_command = torch.cat([self.joint_pos[ids], self.joint_vel[ids]], dim=-1)
        clean_pos = self.anchor_pos_w[ids]
        clean_quat = self.anchor_quat_w[ids]
        self._command_history[ids] = clean_command[:, None, :]
        self._anchor_pos_history[ids] = clean_pos[:, None, :]
        self._anchor_quat_history[ids] = clean_quat[:, None, :]
        self._history_cursor[ids] = 0
        self._episode_reference_step[ids] = 0
        self._last_clean_step[ids] = self.time_steps[ids]
        self._policy_command[ids] = clean_command
        self._policy_anchor_pos_w[ids] = clean_pos
        self._policy_anchor_quat_w[ids] = clean_quat
        self._policy_source_time_step[ids] = self.time_steps[ids]
        self._frozen_command[ids] = clean_command
        self._frozen_anchor_pos_w[ids] = clean_pos
        self._frozen_anchor_quat_w[ids] = clean_quat
        self._frozen_source_time_step[ids] = self.time_steps[ids]

    def _update_policy_reference(self) -> None:
        # A discontinuity means an environment was reset or a deterministic
        # evaluator rewound the reference.  Bootstrap only those environments.
        expected = self._last_clean_step + 1
        reset_ids = torch.nonzero(self.time_steps != expected, as_tuple=False).flatten()
        if reset_ids.numel():
            self._bootstrap_policy_history(reset_ids)
        continuing = torch.nonzero(self.time_steps == expected, as_tuple=False).flatten()
        if not continuing.numel():
            return

        ids = continuing
        self._episode_reference_step[ids] += 1
        self._history_cursor[ids] = (self._history_cursor[ids] + 1) % self._history_size
        cursor = self._history_cursor[ids]
        self._command_history[ids, cursor] = torch.cat([self.joint_pos[ids], self.joint_vel[ids]], dim=-1)
        self._anchor_pos_history[ids, cursor] = self.anchor_pos_w[ids]
        self._anchor_quat_history[ids, cursor] = self.anchor_quat_w[ids]
        delayed_cursor = (cursor - int(self.cfg.reference_delay_steps)) % self._history_size
        delayed_command = self._command_history[ids, delayed_cursor]
        delayed_pos = self._anchor_pos_history[ids, delayed_cursor]
        delayed_quat = self._anchor_quat_history[ids, delayed_cursor]

        episode_step = self._episode_reference_step[ids]
        freeze_start = int(self.cfg.reference_freeze_start_step)
        freeze_end = freeze_start + int(self.cfg.reference_freeze_steps)
        freezing = (episode_step >= freeze_start) & (episode_step < freeze_end)
        starts = ids[episode_step == freeze_start]
        if starts.numel():
            # Capture the immediately preceding *delayed* stream frame, then
            # reuse it for exactly L policy frames.
            self._frozen_command[starts] = self._policy_command[starts]
            self._frozen_anchor_pos_w[starts] = self._policy_anchor_pos_w[starts]
            self._frozen_anchor_quat_w[starts] = self._policy_anchor_quat_w[starts]
            self._frozen_source_time_step[starts] = self._policy_source_time_step[starts]
        normal = ~freezing
        if normal.any():
            normal_ids = ids[normal]
            self._policy_command[normal_ids] = delayed_command[normal]
            self._policy_anchor_pos_w[normal_ids] = delayed_pos[normal]
            self._policy_anchor_quat_w[normal_ids] = delayed_quat[normal]
            age = torch.minimum(
                torch.full_like(episode_step[normal], int(self.cfg.reference_delay_steps)), episode_step[normal]
            )
            self._policy_source_time_step[normal_ids] = self.time_steps[normal_ids] - age
        frozen_ids = ids[freezing]
        if frozen_ids.numel():
            self._policy_command[frozen_ids] = self._frozen_command[frozen_ids]
            self._policy_anchor_pos_w[frozen_ids] = self._frozen_anchor_pos_w[frozen_ids]
            self._policy_anchor_quat_w[frozen_ids] = self._frozen_anchor_quat_w[frozen_ids]
            self._policy_source_time_step[frozen_ids] = self._frozen_source_time_step[frozen_ids]
        self._last_clean_step[ids] = self.time_steps[ids]

    def _update_command(self):
        super()._update_command()
        self._update_policy_reference()

    @property
    def policy_command(self) -> torch.Tensor:
        return self._policy_command

    @property
    def policy_anchor_pos_w(self) -> torch.Tensor:
        return self._policy_anchor_pos_w

    @property
    def policy_anchor_quat_w(self) -> torch.Tensor:
        return self._policy_anchor_quat_w

    @property
    def policy_source_time_step(self) -> torch.Tensor:
        """Clean motion frame represented by the faulted policy reference."""
        return self._policy_source_time_step


@configclass
class ReferenceFaultMotionCommandCfg(MotionCommandCfg):
    class_type: type = ReferenceFaultMotionCommand
    reference_delay_steps: int = 0
    reference_freeze_steps: int = 0
    reference_freeze_start_step: int = 100  # 2 seconds at 50 Hz
