"""Stable contracts shared by pipeline stages."""

EXPECTED_JOINTS = 29
EXPECTED_BODIES = 30
EXPECTED_FPS = 50.0

# Different existing WBT exporters have used different field names. The
# validator reports the actual schema and recognises these aliases.
MOTION_FIELD_ALIASES = {
    "joint_positions": ("dof_pos", "dps", "joint_pos", "joint_positions"),
    "root_positions": ("root_pos", "gts", "root_positions", "body_pos_w"),
    "root_rotations": ("root_rot", "grs", "root_rotations", "body_quat_w"),
}

WBT_REQUIRED_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
