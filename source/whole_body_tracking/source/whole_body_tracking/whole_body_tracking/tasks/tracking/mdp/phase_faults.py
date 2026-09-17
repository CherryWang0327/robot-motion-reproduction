"""Windowed policy-reference phase faults; clean motion state is untouched."""
from __future__ import annotations

import torch
from isaaclab.utils import configclass

from .reference_faults import ReferenceFaultMotionCommand, ReferenceFaultMotionCommandCfg


class PhaseFaultMotionCommand(ReferenceFaultMotionCommand):
    """Separate evaluation command supporting lag, lead, freeze, and jitter."""

    def _update_policy_reference(self):
        expected = self._last_clean_step + 1
        reset_ids = torch.nonzero(self.time_steps != expected, as_tuple=False).flatten()
        if reset_ids.numel():
            self._bootstrap_policy_history(reset_ids)
        ids = torch.nonzero(self.time_steps == expected, as_tuple=False).flatten()
        if not ids.numel():
            return
        self._episode_reference_step[ids] += 1
        self._history_cursor[ids] = (self._history_cursor[ids] + 1) % self._history_size
        cursor = self._history_cursor[ids]
        clean_command = torch.cat([self.joint_pos[ids], self.joint_vel[ids]], dim=-1)
        self._command_history[ids, cursor] = clean_command
        self._anchor_pos_history[ids, cursor] = self.anchor_pos_w[ids]
        self._anchor_quat_history[ids, cursor] = self.anchor_quat_w[ids]

        step = self._episode_reference_step[ids]
        active = (step >= int(self.cfg.phase_fault_start_step)) & (
            step < int(self.cfg.phase_fault_start_step) + int(self.cfg.phase_fault_duration_steps)
        )
        # Default: current clean reference.  These properties are never used by
        # reward, termination, metrics, reset, or simulator initialization.
        self._policy_command[ids] = clean_command
        self._policy_anchor_pos_w[ids] = self.anchor_pos_w[ids]
        self._policy_anchor_quat_w[ids] = self.anchor_quat_w[ids]
        self._policy_source_time_step[ids] = self.time_steps[ids]

        if active.any() and self.cfg.phase_fault_mode != "clean":
            fault_ids, fault_cursor, fault_step = ids[active], cursor[active], step[active]
            mode = self.cfg.phase_fault_mode
            if mode == "freeze":
                starts = fault_ids[fault_step == int(self.cfg.phase_fault_start_step)]
                if starts.numel():
                    self._frozen_command[starts] = self._policy_command[starts]
                    self._frozen_anchor_pos_w[starts] = self._policy_anchor_pos_w[starts]
                    self._frozen_anchor_quat_w[starts] = self._policy_anchor_quat_w[starts]
                    self._frozen_source_time_step[starts] = self._policy_source_time_step[starts]
                self._policy_command[fault_ids] = self._frozen_command[fault_ids]
                self._policy_anchor_pos_w[fault_ids] = self._frozen_anchor_pos_w[fault_ids]
                self._policy_anchor_quat_w[fault_ids] = self._frozen_anchor_quat_w[fault_ids]
                self._policy_source_time_step[fault_ids] = self._frozen_source_time_step[fault_ids]
            else:
                if mode == "jitter_delay":
                    # Stateless deterministic per-environment pseudo-random delay.
                    age = (fault_step * 1103515245 + fault_ids * 12345 + int(self.cfg.phase_fault_seed)) % 11
                else:
                    age = torch.full_like(fault_step, int(self.cfg.phase_fault_magnitude_steps))
                if mode == "phase_lead":
                    source = torch.clamp(self.time_steps[fault_ids] + age, max=self.motion.time_step_total - 1)
                    self._policy_command[fault_ids] = torch.cat(
                        [self.motion.joint_pos[source], self.motion.joint_vel[source]], dim=-1
                    )
                    self._policy_anchor_pos_w[fault_ids] = (
                        self.motion.body_pos_w[source, self.motion_anchor_body_index] + self._env.scene.env_origins[fault_ids]
                    )
                    self._policy_anchor_quat_w[fault_ids] = self.motion.body_quat_w[source, self.motion_anchor_body_index]
                    self._policy_source_time_step[fault_ids] = source
                else:  # phase_lag or jitter_delay
                    history_cursor = (fault_cursor - age) % self._history_size
                    self._policy_command[fault_ids] = self._command_history[fault_ids, history_cursor]
                    self._policy_anchor_pos_w[fault_ids] = self._anchor_pos_history[fault_ids, history_cursor]
                    self._policy_anchor_quat_w[fault_ids] = self._anchor_quat_history[fault_ids, history_cursor]
                    age = torch.minimum(age, fault_step)
                    self._policy_source_time_step[fault_ids] = self.time_steps[fault_ids] - age
        self._last_clean_step[ids] = self.time_steps[ids]


@configclass
class PhaseFaultMotionCommandCfg(ReferenceFaultMotionCommandCfg):
    class_type: type = PhaseFaultMotionCommand
    phase_fault_mode: str = "clean"
    phase_fault_start_step: int = 0
    phase_fault_duration_steps: int = 10
    phase_fault_magnitude_steps: int = 20
    phase_fault_seed: int = 0
