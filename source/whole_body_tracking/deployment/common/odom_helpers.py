"""Small, source-agnostic helpers for the obs_run Odometry2 contract."""

import numpy as np
from scipy.spatial.transform import Rotation


def velocity_odom_to_body(velocity_odom, quaternion_xyzw):
    """Rotate an odometry/world-frame velocity into the reported body frame."""
    rotation = Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=np.float64))
    return rotation.inv().apply(np.asarray(velocity_odom, dtype=np.float64))
