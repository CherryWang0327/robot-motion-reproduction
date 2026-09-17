import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    rz_waist = R.from_euler("z", waist_yaw).as_matrix()
    r_torso = R.from_quat(
        [imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]
    ).as_matrix()
    r_pelvis = np.dot(r_torso, rz_waist.T)
    w = np.dot(rz_waist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(r_pelvis).as_quat()[[3, 0, 1, 2]], w


def transform_pelvis_to_torso_complete(
    waist_yaw, waist_roll, waist_pitch, pelvis_quat
):
    r_waist_yaw = R.from_euler("z", waist_yaw)
    r_waist_roll = R.from_euler("x", waist_roll)
    r_waist_pitch = R.from_euler("y", waist_pitch)
    r_pelvis = R.from_quat(
        [pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]
    )
    r_torso = r_pelvis * r_waist_yaw * r_waist_roll * r_waist_pitch
    torso_quat_scipy = r_torso.as_quat()
    return np.array(
        [
            torso_quat_scipy[3],
            torso_quat_scipy[0],
            torso_quat_scipy[1],
            torso_quat_scipy[2],
        ]
    )


def transform_pelvis_angular_velocity_to_torso_complete(
    waist_yaw,
    waist_roll,
    waist_pitch,
    waist_yaw_velocity,
    waist_roll_velocity,
    waist_pitch_velocity,
    pelvis_omega,
):
    """Return torso angular velocity expressed in torso coordinates."""
    r_yaw = R.from_euler("z", waist_yaw).as_matrix()
    r_roll = R.from_euler("x", waist_roll).as_matrix()
    r_pitch = R.from_euler("y", waist_pitch).as_matrix()
    r_pelvis_torso = r_yaw @ r_roll @ r_pitch
    omega_relative_pelvis = (
        np.array([0.0, 0.0, waist_yaw_velocity])
        + r_yaw @ np.array([waist_roll_velocity, 0.0, 0.0])
        + r_yaw @ r_roll @ np.array([0.0, waist_pitch_velocity, 0.0])
    )
    omega_total_pelvis = np.asarray(pelvis_omega, dtype=np.float64).reshape(3) + omega_relative_pelvis
    return r_pelvis_torso.T @ omega_total_pelvis
