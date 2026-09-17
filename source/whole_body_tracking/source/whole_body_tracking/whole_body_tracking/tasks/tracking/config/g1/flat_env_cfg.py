from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.mdp.reference_faults import ReferenceFaultMotionCommandCfg
from whole_body_tracking.tasks.tracking.mdp.phase_faults import PhaseFaultMotionCommandCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


@configclass
class G1FlatRobustEnvCfg(G1FlatEnvCfg):
    """Independent robustness-training task for the Taichi sim2sim failure envelope.

    The default task stays unchanged.  This variant exposes policies to the
    low-friction edge observed in MuJoCo and to larger recovery velocities.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.6)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.2)
        self.events.push_robot.interval_range_s = (0.8, 2.5)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.8, 0.8),
            "y": (-0.8, 0.8),
            "z": (-0.25, 0.25),
            "roll": (-0.70, 0.70),
            "pitch": (-0.70, 0.70),
            "yaw": (-1.0, 1.0),
        }


@configclass
class G1FlatRefFaultEvalEnvCfg(G1FlatEnvCfg):
    """Isolated evaluation-only variant; clean command semantics are unchanged."""

    def __post_init__(self):
        super().__post_init__()
        # Preserve every base command setting and replace only the command class.
        self.commands.motion.class_type = ReferenceFaultMotionCommandCfg.class_type
        self.commands.motion.reference_delay_steps = 0
        self.commands.motion.reference_freeze_steps = 0
        self.commands.motion.reference_freeze_start_step = 100
        self.observations.policy.command.func = mdp.faulted_generated_commands
        self.observations.policy.motion_anchor_pos_b.func = mdp.faulted_motion_anchor_pos_b
        self.observations.policy.motion_anchor_ori_b.func = mdp.faulted_motion_anchor_ori_b


@configclass
class G1FlatPhaseFaultEvalEnvCfg(G1FlatEnvCfg):
    """Independent task for short, high-dynamics phase faults only."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = PhaseFaultMotionCommandCfg.class_type
        # Allocate enough history for the largest evaluated lag/jitter (30).
        self.commands.motion.reference_delay_steps = 30
        self.commands.motion.reference_freeze_steps = 0
        self.commands.motion.phase_fault_mode = "clean"
        self.commands.motion.phase_fault_start_step = 0
        self.commands.motion.phase_fault_duration_steps = 10
        self.commands.motion.phase_fault_magnitude_steps = 20
        self.commands.motion.phase_fault_seed = 0
        self.observations.policy.command.func = mdp.faulted_generated_commands
        self.observations.policy.motion_anchor_pos_b.func = mdp.faulted_motion_anchor_pos_b
        self.observations.policy.motion_anchor_ori_b.func = mdp.faulted_motion_anchor_ori_b
